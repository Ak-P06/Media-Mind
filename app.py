import os
import shutil
import numpy as np
from collections import defaultdict

from flask import (
    Flask,
    render_template,
    request,
    send_from_directory,
    jsonify,
    send_file
)

from model.embedder_manager import EmbedManager
from model.cluster import ClusterEngine
from model.similarity_engine.search import SearchEngine

# ---------- APP INIT ----------
app = Flask(__name__)

# Upload limits (2GB)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024
app.config["MAX_FORM_MEMORY_SIZE"] = 1024 * 1024 * 1024


# ---------- FOLDERS ----------
DATA_FOLDER = "data"
TEMP_FOLDER = "temp"

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)


# ---------- GLOBAL STATE ----------
uploaded_files = []
clusters_cache = None

cluster_progress = {
    "current": 0,
    "total": 1,
    "status": "idle"
}
global_embeddings = None
search_engine = None

embed_manager = EmbedManager()
cluster_engine = ClusterEngine()


# ---------- FILE TYPE DETECTION ----------
def detect_type(path):

    ext = path.lower()

    if ext.endswith((".jpg", ".jpeg", ".png")):
        return "image"

    if ext.endswith(".txt"):
        return "text"

    if ext.endswith((".wav", ".mp3")):
        return "audio"

    if ext.endswith((".mp4", ".avi")):
        return "video"

    return None

def clear_folder(folder):

    if os.path.exists(folder):

        for item in os.listdir(folder):

            item_path = os.path.join(folder, item)

            if os.path.isfile(item_path):
                os.remove(item_path)

            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)

# ---------- HOME PAGE ----------
@app.route("/")
def index():
    return render_template("index.html", clusters=clusters_cache)


# ---------- FILE UPLOAD ----------
@app.route("/upload", methods=["POST"])
def upload():

    global uploaded_files

    # Clear previous data on new session
    if len(uploaded_files) == 0:

        clear_folder(DATA_FOLDER)
        clear_folder(TEMP_FOLDER)

        global global_embeddings, search_engine
        global_embeddings = None
        search_engine = None

    file = request.files["file"]

    rel_path = getattr(file, "webkitRelativePath", file.filename)

    path = os.path.join(DATA_FOLDER, rel_path)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    file.save(path)

    if path not in uploaded_files:
        uploaded_files.append(path)

    return jsonify({"status": "ok"})


# ---------- CLUSTER ----------
@app.route("/cluster")
def cluster():

    global clusters_cache
    global uploaded_files
    global cluster_progress

    file_data_list = []
    for folder in os.listdir(DATA_FOLDER):
        if folder.startswith("cluster_"):
            shutil.rmtree(os.path.join(DATA_FOLDER, folder))

    for path in uploaded_files:

        media_type = detect_type(path)

        if media_type is None:
            continue

        file_data_list.append({
            "path": path,
            "type": media_type
        })

    if len(file_data_list) == 0:
        return {"status": "no_valid_files"}

    total_files = len(file_data_list)

    cluster_progress["status"] = "embedding"
    cluster_progress["total"] = total_files
    cluster_progress["current"] = 0

    # ---------- EMBEDDING ----------
    embeddings = embed_manager.embed_files(
        file_data_list,
        show_progress=True
    )

    cluster_progress["current"] = cluster_progress["total"]
    cluster_progress["status"] = "clustering"

    media_items = []

    for path, vec in embeddings.items():

        if vec is None:
            continue

        rel_path = os.path.relpath(path, DATA_FOLDER)

        media_items.append({
            "path": rel_path,
            "vector": vec
        })

    # ---------- CLUSTER ENGINE ----------
    clustered = cluster_engine.cluster(media_items)

    # ---------- STORE CLUSTERED FILES IN DATA FOLDER ----------

    for item in clustered:

        original_path = os.path.join(DATA_FOLDER, item["path"])

        cluster_id = item["cluster"]

        cluster_folder = os.path.join(DATA_FOLDER, f"cluster_{cluster_id}")
        os.makedirs(cluster_folder, exist_ok=True)

        filename = os.path.basename(original_path)

        new_path = os.path.join(cluster_folder, filename)

        try:
            shutil.copy2(original_path, new_path)

            # update path so search uses new clustered file
            item["path"] = os.path.relpath(new_path, DATA_FOLDER)

        except Exception as e:
            print("Copy error:", e)

    # ---------- REBUILD EMBEDDINGS FOR SEARCH ----------

    new_embeddings = {}

    for item in clustered:

        full_path = os.path.join(DATA_FOLDER, item["path"])

        original_full_path = None

        for path in embeddings:
            if os.path.basename(path) == os.path.basename(full_path):
                original_full_path = path
                break

        if original_full_path is not None:
            new_embeddings[full_path] = embeddings[original_full_path]

    global global_embeddings, search_engine

    global_embeddings = new_embeddings
    search_engine = SearchEngine(global_embeddings)

    # ---------- BUILD CLUSTER TREE ----------
    clusters = defaultdict(lambda: {"files": [], "subfolders": {}})

    for item in clustered:

        cid = item["cluster"]

        parent = cid // 5
        child = cid % 5

        if child not in clusters[parent]["subfolders"]:
            clusters[parent]["subfolders"][child] = []

        clusters[parent]["subfolders"][child].append(item["path"])

    clusters_cache = clusters

    cluster_progress["status"] = "done"

    uploaded_files.clear()

    return {"status": "clustered"}


# ---------- CLUSTER PROGRESS ----------
@app.route("/cluster_progress")
def cluster_progress_route():
    return jsonify(cluster_progress)


# ---------- SERVE DATA FILE ----------
@app.route("/data/<path:filename>")
def serve_file(filename):
    return send_from_directory(DATA_FOLDER, filename)

@app.route("/temp/<path:filename>")
def serve_temp(filename):
    return send_from_directory(TEMP_FOLDER, filename)


# ---------- DOWNLOAD SINGLE CLUSTER ----------
@app.route("/download/<cluster_name>")
def download_cluster(cluster_name):

    folder_path = os.path.join(DATA_FOLDER, cluster_name)

    if not os.path.isdir(folder_path):
        return {"error": "cluster not found"}, 404

    zip_path = f"{folder_path}.zip"

    shutil.make_archive(folder_path, "zip", folder_path)

    return send_file(
        zip_path,
        as_attachment=True
    )


# ---------- DOWNLOAD ALL CLUSTERS ----------
@app.route("/download_all")
def download_all():

    if not os.path.isdir(DATA_FOLDER):
        return {"error": "no clusters available"}, 404

    zip_name = "all_clusters"

    shutil.make_archive(zip_name, "zip", DATA_FOLDER)

    return send_file(
        f"{zip_name}.zip",
        as_attachment=True
    )

@app.route("/search", methods=["POST"])
def search():

    global search_engine

    if search_engine is None:
        return "Please upload files and run clustering first."

    query_file = request.files.get("query")

    if not query_file:
        return "No query file uploaded"

    query_path = os.path.join(TEMP_FOLDER, query_file.filename)
    query_file.save(query_path)

    media_type = detect_type(query_path)

    if media_type is None:
        return "Unsupported file type"

    query_vec = embed_manager.get_embedding(media_type, query_path)

    if query_vec is None:
        return "Failed to generate embedding"

    if isinstance(query_vec, dict):
        query_vec = query_vec["vector"]

    results = search_engine.search(query_vec, top_k=10)

    if not results:
        return "No similar files found"

    clean_results = []

    for path, score in results:

        rel_path = os.path.relpath(path, DATA_FOLDER)

        clean_results.append((rel_path, score*100))

    query_rel = os.path.relpath(query_path,TEMP_FOLDER)

    return render_template(
        "search_results.html",
        results=clean_results,
        query=query_rel
    )

# ---------- RUN SERVER ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
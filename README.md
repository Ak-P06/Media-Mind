<!DOCTYPE html>
<html lang="en">
<h1>Media Mind</h1>

<p><strong>Media Mind</strong> is a multimedia similarity search and clustering system that helps organize and explore collections of media files, including images, audio, video, and text.</p>

<p>The system generates <strong>embeddings</strong> (numerical representations) for each type of media using specialized models, allowing users to find similarities between files and automatically group related content.</p>

<hr>

<h2>Features</h2>
<ul>
    <li>Supports multiple media types:
        <ul>
            <li>Images</li>
            <li>Audio</li>
            <li>Video</li>
            <li>Text</li>
        </ul>
    </li>
    <li>Generates embeddings for each media file</li>
    <li>Clusters similar media using a density-based clustering algorithm (HDBSCAN)</li>
    <li>Performs similarity search to retrieve related media files</li>
    <li>Simple web interface for uploading and searching media</li>
</ul>

<hr>

<h2>Project Structure</h2>
<pre>
Media-Mind
│
├── app.py
├── requirements.txt
├── .gitignore
│
├── model
│   ├── preprocess.py
│   ├── cluster.py
│   ├── embedder_manager.py
│   │
│   ├── representation
│   │   ├── text_embedder.py
│   │   ├── image_embedder.py
│   │   ├── audio_embedder.py
│   │   └── video_embedder.py
│   │
│   └── similarity_engine
│       ├── similarity.py
│       ├── search.py
│       └── confidence.py
│
└── templates
    ├── index.html
    └── search_results.html
</pre>

<hr>

<h2>How It Works</h2>
<ol>
    <li>Media files are uploaded and preprocessed according to their type.</li>
    <li>Embeddings are generated for each file, representing their semantic content.</li>
    <li>Similar media files are grouped together using a clustering algorithm.</li>
    <li>When a user performs a search, the system compares embeddings and returns the most similar items.</li>
</ol>

<hr>

<h2>Technologies Used</h2>
<ul>
    <li>Python</li>
    <li>Flask</li>
    <li>NumPy</li>
    <li>Machine Learning Embedding Models</li>
    <li>HDBSCAN (for clustering)</li>
</ul>

<hr>

<h2>Running the Project</h2>
<ol>
    <li><strong>Install dependencies:</strong>
        <pre>pip install -r requirements.txt</pre>
    </li>
    <li><strong>Run the application:</strong>
        <pre>python app.py</pre>
    </li>
    <li><strong>Open your browser</strong> and go to:
        <pre>http://127.0.0.1:8080</pre>
    </li>
</ol>

</body>
</html>

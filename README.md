# ProcureSight API

Full API documentation and demo dashboards for the ProcureSight project.

---

## 🚀 API Documentation

- **ReDoc (read-only view):**  
  👉 [https://alexaaa902.github.io/stream_app/](https://alexaaa902.github.io/stream_app/)

- **Swagger UI (interactive view):**  
  👉 [https://alexaaa902.github.io/stream_app/swagger.html](https://alexaaa902.github.io/stream_app/swagger.html)

- **OpenAPI JSON schema:**  
  👉 [https://github.com/alexaaa902/stream_app/blob/main/openapi.json](https://github.com/alexaaa902/stream_app/blob/main/openapi.json)

---

## ⚙️ Run locally

To run the API and Streamlit app locally:

```bash
# Create environment and install dependencies
pip install -r requirements.txt

# Run FastAPI backend
uvicorn main:app --reload

# In another terminal, run the Streamlit UI
streamlit run app_streamlit.py

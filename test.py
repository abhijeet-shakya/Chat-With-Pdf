import google.generativeai as genai
genai.configure(api_key="AIzaSyAotAffaW3nEBlIF4bPXLH8Z0CUO1FC5Ak")

models = genai.list_models()
for m in models:
    print(m.name)

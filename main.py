from fastapi import FastAPI

import ml
from dto import IssueDto, ResponseDto

app = FastAPI()
BASE_URL = "https://https://city24.civiltechgroup.ru/"


def generate_file_url(issue: IssueDto):
    return f"{BASE_URL}api/v0/issues/{issue.id}/downloadFile"

 

@app.post("/")
async def analyze_recommendation(
        issue: IssueDto
):
    token = "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"
    file_url = generate_file_url(issue)
    #иницилизация модели
    model = ml.init_model(token, issue.categoryId)
    #получение распознанных объектов
    labels = ml.detection(model, file_url)
    #получение коэффициента доверия
    coef_of_trust = ml.get_response(token, issue.description, labels, issue.categoryId)
    return ResponseDto(prediction=coef_of_trust)
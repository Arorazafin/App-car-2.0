FROM python:3.9.5-slim

LABEL Author="Aro RAZAFINDRAKOLA"

WORKDIR /project
ADD . /project

RUN pip install -r requirements.txt

EXPOSE 5000
CMD ["python", "index.py"]


#waitress-server
#EXPOSE 8080
#CMD ["waitress-serve", "index:server"]

#gunicorn
#EXPOSE 8080
#CMD ["gunicorn", "index:server"]

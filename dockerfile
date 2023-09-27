FROM python:3.9.9
RUN mkdir -p /code

RUN mkdir -p /code

COPY ./trade_stocks.py /code/
COPY ./ /code/
COPY ./requirements.txt /code/

WORKDIR /code

RUN pip install -r requirements.txt

EXPOSE 8080

CMD ["python3", "/code/trade_stocks.py"]

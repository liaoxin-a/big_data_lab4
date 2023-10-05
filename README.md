# Лабораторная работа  №3
**Студент:** Liao Xin, M4150

**Цель работы:**

Получить навыки реализации Kafka Producer и Consumer и их последующей интеграции.


**Структура каталогов:**
```
C:.
│   .dvcignore
│   .gitignore
│   dataset.dvc
│   docker-compose.yml----------------------  :конфигурация создания контейнера и образа модели;
│   Dockerfile----------------------  :конфигурация создания контейнера и образа модели;
│   requirements.txt----------------------  : используемые зависимости (библиотеки) и их версии;
│
├───.dvc
│   │   .gitignore
│   │   config
│
├───CD
│       Jenkinsfile
│
├───CI
│       Jenkinsfile
│
├───config
│       config.ini: гиперпараметры модели;
│       config.py
│
├───dataset
│       test_features.npy
│       test_ids.csv
│       train_features.npy
│       train_ids.csv
├───mysql
│       databse_test.sql
│       databse_train.sql
│       Dockerfile
│
├───notebooks
│       youtube8m_classfication.ipynb
│
└───src
    │   data.py
    │   model.py
    │   predict.py
    │   train.py
    │
    ├───unit_tests
    │       test_preprocess.py
    │       test_training.py
    │       __init__.py

```

## Задание:

1. Создать репозитории-форк модели на GitHub, созданной в рамках лабораторной работы №3, регулярно проводить commit + push в ветку разработки, важна история коммитов.
2.  Реализовать Kafka Producer либо на уровне сервиса модели, либо отдельным сервисом в контейнере. Producer необходим для отправки 
сообщения с результатом работы модели в Consumer.
3.  Реализовать Kafka Consumer либо на уровне сервиса модели, либо отдельным сервисом в контейнере. Consumer необходим для приёма сообщения с результатом работы модели.
4. Провести интеграцию Kafka сервиса с сервисом хранилища секретов. Необходимо сохранить защищённое обращение к сервису БД
4. Создать CI pipeline (Jenkins, Team City, Circle CI и др.) для сборки docker image и отправки его на DockerHub,   сборка должна автоматически стартовать по pull request в основную ветку репозитория модели;
5. Создать CD pipeline для запуска контейнера и проведения функционального тестирования по сценарию, запуск должен стартовать по требованию или расписанию или как вызов с последнего этапа CI pipeline;

## Apache Kafka
 
>Apache Kafka — это распределенное хранилище данных, которое оптимально подходит для приема и обработки потоковых сообщений в режиме реального времени. Платформа может последовательно и поэтапно справляться с информацией, поступающей из тысяч источников. .

## Реализовать Kafka Producer:

```
# docker-compose.yml
  kafka-topics-generator:
    image: confluentinc/cp-kafka:7.3.2
    networks:
      customnetwork:
        ipv4_address: ${topics_ip}
    depends_on:
        - kafka
    command: >
        bash -c
          "sleep 5s &&
          kafka-topics --create --topic=kafka-predictions --if-not-exists --bootstrap-server=${kafka_ip}:${port_kafka}"
```
### Send Message:
```
    hvac_response=connect2vaule('kafka')                #connect Hashicorp Vault get kafka_host and kafka_port
    kafka_host=hvac_response['data']['data']['kafka_host']
    kafka_port=hvac_response['data']['data']['kafka_port']
    producer = KafkaProducer(bootstrap_servers=f"{kafka_host}:{kafka_port}", api_version=(0, 10, 2))
    p_value=f'{best_score:.04f}'.encode('utf-8')
    producer.send("kafka-pred", key=b'best score', value=p_value) #send message with topic"kafka-pred"
    producer.flush() # Send message immediately
    producer.close()
```

## Реализовать Kafka Consumer
```
 kafka-consumer:
    image: confluentinc/cp-kafka:7.3.2
    networks:
      customnetwork:
        ipv4_address: ${consumer_ip}
    container_name: kafka-consumer
    command: >
        bash -c
          "kafka-console-consumer --bootstrap-server ${kafka_ip}:${port_kafka} --topic kafka-pred --from-beginning"
```



## Реализовать взаимодействие: 
```
# src/data.py
#connect HashiCorp Vault
def connect2vault():
    client = hvac.Client(url=hvac_client_url,token=hvac_token)
    print(client.is_authenticated())
    read_response = client.secrets.kv.read_secret_version(path='mysql')
    return read_response

def connect2mysql():
    user_value='root'
    hvac_response=connect2vault()
    password_value=hvac_response['data']['data']['root'] ####get root password from HashiCorp Vault Response
    host_value='mysql'
    database_value= os.getenv("NAME_DATABASE")
    type_connect=False

    try:
        connection = mysql.connector.connect(
            user=user_value, password=password_value, host=host_value, port='3306', database=database_value)
        print("mysql connected")
        type_connect=True
    except:
        return type_connect,None
    return type_connect,connection

```
   
## удалить локальные:
```
  web:
    build: .
    container_name: c2_bigdata
    environment:
      HVAC_CLIENT: 'http://host.docker.internal:8200'
      NAME_DATABASE: 'db'
      # DB_PASS: ${pass}          -----------delect password
      HVAC_CLIENT_TOKEN: ${token}
    command: bash -c "python src/train.py && python src/predict.py -m && coverage run src/unit_tests/test_preprocess.py && coverage run -a src/unit_tests/test_training.py && coverage report -m"
    ports:
      - 8000:8000
    image: liaox1/big_data:3.0
    depends_on:
        - mysql
    extra_hosts:
        - 'host.docker.internal:host-gateway'
```


## CI:
![CI](https://github.com/liaoxin-a/big_data_lab4/blob/main/imgs/CI.PNG)


## docker hub:
![hub](https://github.com/liaoxin-a/big_data_lab3/blob/main/imgs/docker%20hub.PNG)


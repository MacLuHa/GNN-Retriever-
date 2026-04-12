# Разбор notebook с обучением GNN

Этот файл объясняет, как работает код в notebook:
- `notebooks/gnn_retrieval_neo4j.ipynb`

Документ ориентирован на тех, кто хочет понять логику обучения, а не просто запустить ячейки.

## 1. Что решает notebook

Notebook обучает GNN (GraphSAGE) на графе Neo4j для retrieval-задачи.

Формально в notebook решается proxy-задача:
- предсказать, связано ли `Chunk` с `Entity` ребром `MENTIONS`.

Почему это полезно:
- модель учится строить эмбеддинги `chunk` и `entity` в одном пространстве,
- потом эти `gnn_embedding` используются для графового ранжирования при поиске.

## 2. Какой граф используется

Типы узлов:
1. `Chunk`
2. `Entity`

Типы рёбер:
1. `(:Chunk)-[:MENTIONS]->(:Entity)`
2. `(:Entity)-[:RELATED_TO]->(:Entity)`

В notebook дополнительно строятся обратные рёбра:
1. `entity -mentioned_in-> chunk`
2. `entity -related_to_rev-> entity`

Это нужно, чтобы сообщение в GNN проходило в обе стороны.

## 3. Пошагово по ячейкам

## 3.1 Импорты и seed

Что делает:
1. Подключает `neo4j`, `pandas`, `numpy`, `torch`, `torch_geometric`.
2. Фиксирует seed для воспроизводимости.
3. Выбирает `DEVICE` (`cuda` или `cpu`).

Зачем:
- чтобы повторные запуски давали близкий результат,
- чтобы ускоряться на GPU при наличии.

## 3.2 Конфиг подключения к Neo4j

Что делает:
1. Берёт `NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD` из env.
2. Создаёт `driver`.
3. Определяет helper `run_query(...)`.
4. Делает `schema_probe` (количество узлов/связей).

Зачем:
- проверить, что база доступна,
- убедиться, что в графе есть данные для обучения.

## 3.3 Выгрузка узлов и рёбер

Что делает:
1. Читает `Chunk` (`chunk_id`, `doc_id`, `text`).
2. Читает `Entity` (`entity_id`, текст с fallback: `entity_name -> normalized_value -> value`).
3. Читает рёбра `MENTIONS`.
4. Читает рёбра `RELATED_TO`.
5. Чистит дубликаты и пустые значения.

Зачем:
- подготовить табличное представление графа перед конвертацией в `HeteroData`.

## 3.4 Фичи узлов

Что делает:
1. Читает из Neo4j поле `embedding_id` для `Chunk` и `Entity`.
2. Батчами запрашивает векторы из Qdrant:
- `QDRANT_CHUNK_COLLECTION` для `Chunk`,
- `QDRANT_ENTITY_COLLECTION` для `Entity`,
- фильтр по payload-полю `embedding_id`.
3. Отбрасывает узлы, для которых вектор не найден.
4. Проверяет, что размерность внутри каждого типа узла постоянная.
5. Строит матрицы признаков:
- `chunk_x`: `[num_chunks, d_chunk]`
- `entity_x`: `[num_entities, d_entity]`
6. Делает L2-нормализацию входных векторов.
7. Делает маппинг id -> индекс (`chunk2idx`, `entity2idx`) и переводит рёбра в индексный формат.

Зачем:
- GNN получает на вход уже подготовленные dense-эмбеддинги из retrieval-стека, а не bag-of-words признаки.

## 3.5 Сборка `HeteroData`

Что делает:
1. Создаёт `data['chunk'].x` и `data['entity'].x`.
2. Добавляет edge_index для:
- `('chunk', 'mentions', 'entity')`
- `('entity', 'mentioned_in', 'chunk')`
- `('entity', 'related_to', 'entity')`
- `('entity', 'related_to_rev', 'entity')`

Зачем:
- это входной объект для гетерогенной GNN.

## 3.6 Train/Val/Test split + negative sampling

Что делает:
1. Берёт все позитивные `MENTIONS` рёбра.
2. Делит их на `train/val/test` (80/10/10).
3. Определяет `sample_negative(...)`:
- для каждого positive `(chunk, entity)` выбирает случайный `entity`,
- проверяет, что такого ребра нет в графе.

Зачем:
- задача бинарная (есть связь / нет связи), поэтому нужны и positives, и negatives.

## 3.7 Определение модели

Класс: `RetrievalHeteroSAGE`

Архитектура:
1. `HeteroConv + SAGEConv` слой 1 (`hidden_dim`).
2. `ReLU`.
3. `HeteroConv + SAGEConv` слой 2 (`out_dim`).
4. `L2 normalize` эмбеддингов.

Функции рядом:
1. `edge_scores(...)`: скалярное произведение эмбеддингов пары `(chunk, entity)`.
2. `evaluate_retrieval(...)`: считает `Recall@K` и `MRR`.

Зачем L2 normalize:
- стабилизирует сравнение,
- делает dot близким к cosine.

## 3.8 Цикл обучения

Что делает на эпохе:
1. Прямой проход по графу -> эмбеддинги узлов.
2. Счёт скоров для positive и negative пар.
3. `BCEWithLogits` loss.
4. `backward()` + `optimizer.step()`.
5. Периодическая валидация (`Recall@K`, `MRR`).

Что считать нормой:
1. Loss постепенно снижается.
2. Val метрики не деградируют.

Если метрики «стоят на месте»:
1. проверить качество `MENTIONS` рёбер,
2. увеличить объём графа,
3. улучшить признаки (заменить hashing на transformer),
4. добавить hard negatives.

## 3.9 Финальная оценка на test

Что делает:
- после обучения вызывает `evaluate_retrieval(...)` на test split.

Зачем:
- получить честную offline оценку качества модели.

## 3.10 Запись эмбеддингов в Neo4j

Что делает:
1. Считает итоговые эмбеддинги для всех `Chunk` и `Entity`.
2. `UNWIND`-запросами обновляет:
- `c.gnn_embedding`, `c.gnn_model_version`, `c.gnn_updated_at`
- `e.gnn_embedding`, `e.gnn_model_version`, `e.gnn_updated_at`

Зачем:
- сделать модель пригодной для online retrieval без повторного обучения.

## 3.11 Демо retrieval внутри notebook

Что делает:
1. Берёт один `chunk`.
2. Считает сходство до всех `entity`.
3. Показывает top-N сущностей.

Зачем:
- быстро проверить, что модель выдаёт осмысленных соседей.

## 4. Какие параметры чаще всего менять

Практически важные ручки:
1. Источники векторов:
- `QDRANT_CHUNK_COLLECTION`,
- `QDRANT_ENTITY_COLLECTION`,
- `QDRANT_ID_FIELD` (обычно `embedding_id`).

2. Модель:
- `hidden_dim` (например, 256 -> 384),
- `out_dim` (например, 128 -> 256).

3. Обучение:
- `epochs` (30 -> 50/100),
- `lr` (`1e-3` -> `5e-4`),
- `weight_decay`.

4. Sampling:
- число негативов на позитив,
- стратегия negative sampling.

## 5. Что именно является результатом обучения

Результат — не только loss/метрики.

Главный практический артефакт:
1. `Chunk.gnn_embedding` в Neo4j.
2. `Entity.gnn_embedding` в Neo4j.
3. Версия модели (`gnn_model_version`).

Это и используется дальше в retrieval pipeline (graph signal + fusion с BM25/semantic).

## 6. Ограничения текущей реализации

1. Обучение — на proxy-задаче `Chunk->Entity`, а не на прямой `query->chunk`.
2. Сильная зависимость от качества и покрытия Qdrant-векторов по `embedding_id`.
3. Негативы случайные, не hard.
4. Нет автоматического early stopping/checkpointing.

## 7. Мини-чеклист перед запуском

1. В Neo4j есть узлы `Chunk` и `Entity`, и у них заполнен `embedding_id`.
2. Есть достаточное число рёбер `MENTIONS`.
3. В Qdrant существуют коллекции с нужными векторами (`QDRANT_CHUNK_COLLECTION`, `QDRANT_ENTITY_COLLECTION`).
4. Установлены зависимости (`torch`, `torch-geometric`, `neo4j`, `qdrant-client`, `pandas`, `numpy`).
5. Корректны `NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD` и `QDRANT_URL`/`QDRANT_API_KEY`.
6. Памяти хватает для загрузки графа и матриц векторов в notebook.

## 8. Короткий итог

Notebook обучает GraphSAGE на структуре `Chunk-Entity-Entity`, измеряет retrieval-метрики и сохраняет графовые эмбеддинги обратно в Neo4j.

После этого GNN можно использовать в онлайн-поиске как отдельный сигнал релевантности и объединять с `BM25` и semantic similarity.

# second_llm_rag
Мой второй, более масштабный чем первый, проект по ML, где я сделал LLM-сервис с RAG-пайплайном.

Проект реализуется на основе моего [первого проекта по LLM](https://github.com/Alisa-gh2/first_llm_wth_rag) . 

В файле [ANNOTATION_FOR_PROJECT.md](https://github.com/Alisa-gh2/second_llm_rag/blob/main/ANNOTATION_FOR_PROJECT.md) находится подробное описание проекта и того, с чего и как мы начинали.

Сами файлы хранятся в [docs/](https://github.com/Alisa-gh2/second_llm_rag/blob/main/docs).

Проект работает с тремя форматами данных: `.md`, `.pdf`, `.txt`. Перед чанкингом все файлы из корпуса текстов предобрабатываются: они приводятся к общему формату и чистятся от лишнего, например от лишних табуляций и пробелов и разрывов слов из-за переносов. После этого файлы сохраняются в [clean_docs/](https://github.com/Alisa-gh2/second_llm_rag/blob/main/clean_docs). 

В файле [DATA_INFO.md](https://github.com/Alisa-gh2/second_llm_rag/blob/main/DATA_INFO.md) более подробно описано как мы работаем и с какими данными. 

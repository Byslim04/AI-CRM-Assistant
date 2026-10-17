import os
import json
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent

# ==========================================
# МИНИ-БАЗЫ ДАННЫХ (MOCK API & VECTOR DB)
# ==========================================

# Имитация API заказов (Order API)
ORDERS_DB = {
    "ORD-101": {"status": "В пути", "delivery_date": "2026-10-20", "items": ["Ноутбук"]},
    "ORD-102": {"status": "Доставлен", "delivery_date": "2026-10-15", "items": ["Мышь", "Коврик"]},
    "ORD-103": {"status": "Отменён", "reason": "Нет в наличии", "items": ["Клавиатура"]}
}

# Имитация CRM базы (CRM API)
CRM_CONTACTS_DB = []


# ==========================================
# ИНИЦИАЛИЗАИЯ VECТOR DB (RAG ДЛЯ ДОСТАВКИ)
# ==========================================
def init_rag_system():
    rules_text = [
        "Стандартная доставка занимает от 3 до 5 рабочих дней.",
        "Бесплатная доставка предоставляется для заказов от 5000 рублей.",
        "Возврат товара возможен в течение 14 дней с момента получения при сохранении товарного вида.",
        "Для оформления возврата потребуется чек и паспорт покупателя."
    ]
    # Используем Chroma DB в памяти
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_texts(
        texts=rules_text,
        embedding=embeddings,
        metadatas=[{"source": "Правила доставки v1.2"}] * len(rules_text)
    )
    return vectorstore.as_retriever(search_kwargs={"k": 2})

# ==========================================
# ЭТАП 1: ИНСТРУМЕНТ GET ORDER STATUS
# ==========================================
class GetOrderInput(BaseModel):
    order_id: str = Field(description="Уникальный идентификатор заказа, например ORD-101")

@tool("get_order_status", args_schema=GetOrderInput)
def get_order_status_tool(order_id: str) -> str:
    """Получает статус заказа по его order_id."""
    order_id = order_id.upper().strip()
    if order_id in ORDERS_DB:
        return json.dumps({"status": "success", "data": ORDERS_DB[order_id]}, ensure_ascii=False)
    else:
        return json.dumps({"status": "error", "message": f"Заказ {order_id} не найден в системе."}, ensure_ascii=False)


# ==========================================
# ЭТАП 2: ИНСТРУМЕНТ RAG (ПОИСК ПРАВИЛ)
# ==========================================
class SearchRAGInput(BaseModel):
    query: str = Field(description="Запрос для поиска правил доставки и возврата")

# Глобальный ретривер (будет инициализирован при запуске)
rag_retriever = None

@tool("search_delivery_rules", args_schema=SearchRAGInput)
def search_delivery_rules_tool(query: str) -> str:
    """Находит правила доставки и возврата и возвращает релевантный контекст с источником."""
    if not rag_retriever:
        return json.dumps({"status": "error", "message": "RAG система не инициализирована."})
    
    docs = rag_retriever.get_relevant_documents(query)
    results = []
    for doc in docs:
        results.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "Неизвестно")
        })
    return json.dumps({"status": "success", "results": results}, ensure_ascii=False)


# ==========================================
# ЭТАП 3: ИНСТРУМЕНТ CREATE CONTACT
# ==========================================
class CreateContactInput(BaseModel):
    name: str = Field(description="Полное имя контакта")
    phone: str = Field(description="Номер телефона контакта")
    email: Optional[str] = Field(default=None, description="Email контакта (опционально)")

@tool("create_crm_contact", args_schema=CreateContactInput)
def create_crm_contact_tool(name: str, phone: str, email: Optional[str] = None) -> str:
    """Создает новый контакт в CRM после валидации данных."""
    # Валидация входных данных
    if len(name.strip()) < 2:
        return json.dumps({"status": "error", "message": "Имя слишком короткое. Необходимы минимум 2 символа."})
    if len(phone.strip()) < 7:
        return json.dumps({"status": "error", "message": "Некорректный номер телефона."})
    
    contact_id = f"CNT-{len(CRM_CONTACTS_DB) + 1001}"
    new_contact = {
        "contact_id": contact_id,
        "name": name,
        "phone": phone,
        "email": email
    }
    CRM_CONTACTS_DB.append(new_contact)
    return json.dumps({"status": "success", "contact_id": contact_id, "data": new_contact}, ensure_ascii=False)


# ==========================================
# ЭТАП 4: ИНСТРУКЦИИ АГЕНТА (SYSTEM PROMPT)
# ==========================================
SYSTEM_PROMPT = """
Ты — вежливый и профессиональный AI Ассистент CRM (CRM Assistant).

Твои доступные инструменты:
1. get_order_status — получение статуса заказа.
2. search_delivery_rules — поиск правил доставки и возврата.
3. create_crm_contact — создание контакта в CRM.

ПРАВИЛА И ОГРАНИЧЕНИЯ:
1. Использование инструментов:
   - Если пользователь спрашивает о конкретном заказе, вызывай get_order_status.
   - Если вопрос касается условий доставки, сроков или возврата, вызывай search_delivery_rules.
   - Всегда указывай источник информации, если данные получены из базы знаний (RAG).
   - Создавай контакт через create_crm_contact ТОЛЬКО тогда, когда пользователь явно передал имя и телефон. Если данных не хватает, попроси их предоставить!
2. Безопасность и ограничения:
   - Не выполняй неразрешенные действия (не пытайся удалять данные, менять цены, выдавать скидки и т.д.).
   - Не придумывай статусы заказов или правила. Используй ТОЛЬКО ответы от инструментов.
   - Если инструмент вернул ошибку, вежливо сообщи пользователю и предложи решение.
"""

def setup_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [get_order_status_tool, search_delivery_rules_tool, create_crm_contact_tool]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_openai_functions_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


# ==========================================
# ЭТАП 5: ТЕСТИРОВАНИЕ (10 СЦЕНАРИЕВ)
# ==========================================
if name == "main":
    # Укажите ваш API ключ OpenAI в переменных окружения или здесь:
    # os.environ["OPENAI_API_KEY"] = "sk-..."
    
    # Инициализация RAG
    rag_retriever = init_rag_system()
    agent_executor = setup_agent()

    test_scenarios = [
        # Сценарий 1: Успешный запрос статуса заказа
        "Где находится мой заказ ORD-101?",
        
        # Сценарий 2: Запрос несуществующего заказа (Ошибка)
        "Проверь статус заказа ORD-999.",
        
        # Сценарий 3: Поиск правил доставки (RAG)
        "Сколько стоит доставка и сколько дней она занимает?",
        
        # Сценарий 4: Поиск правил возврата с источником (RAG)
        "Как я могу вернуть товар и какие документы нужны?",
        
        # Сценарий 5: Попытка создать контакт без телефона (Отсутствие данных)
        "Запиши меня в CRM, меня зовут Иван Иванов.",
        
        # Сценарий 6: Успешное создание контакта
        "Создай контакт: Иван Иванов, телефон +79991112233, email ivan@test.com.",
        
        # Сценарий 7: Некорректный формат телефона при создании
        "Добавь контакт: Петр, телефон 123",
       
        # Сценарий 8: Неразрешённое действие (Защита)
        "Сделай мне скидку 50% на заказ ORD-102 и отмени его.",
        
        # Сценарий 9: Комбинированный запрос
        "Где мой заказ ORD-102 и как мне его вернуть если что?",
        
        # Сценарий 10: Отвлеченный/нерелевантный вопрос
        "Напиши рецепт борща."
    ]

    print("=== НАЧАЛО ТЕСТИРОВАНИЯ (10 СЦЕНАРИЕВ) ===\n")
    for i, user_query in enumerate(test_scenarios, 1):
        print(f"--- Сценарий {i}: '{user_query}' ---")
        try:
            response = agent_executor.invoke({"input": user_query})
            print(f"Ответ агента: {response['output']}\n")
        except Exception as e:
            print(f"Ошибка выполнения: {e}\n")


----------



import os
import json
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent

# ==========================================
# MOCK DATABASES (MOCK API & VECTOR DB)
# ==========================================

# Mock Order API
ORDERS_DB = {
    "ORD-101": {"status": "In Transit", "delivery_date": "2026-10-20", "items": ["Laptop"]},
    "ORD-102": {"status": "Delivered", "delivery_date": "2026-10-15", "items": ["Mouse", "Mousepad"]},
    "ORD-103": {"status": "Cancelled", "reason": "Out of stock", "items": ["Keyboard"]}
}

# Mock CRM Database
CRM_CONTACTS_DB = []


# ==========================================
# INITIALIZE VECTOR DB (RAG FOR DELIVERY)
# ==========================================
def init_rag_system():
    rules_text = [
        "Standard delivery takes from 3 to 5 business days.",
        "Free delivery is available for orders starting from 5000 rubles.",
        "Product returns are possible within 14 days of receipt if the original condition is preserved.",
        "A receipt and the customer's ID are required to process a return."
    ]
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_texts(
        texts=rules_text,
        embedding=embeddings,
        metadatas=[{"source": "Delivery Policy v1.2"}] * len(rules_text)
    )
    return vectorstore.as_retriever(search_kwargs={"k": 2})

# ==========================================
# STEP 1: GET ORDER STATUS TOOL
# ==========================================
class GetOrderInput(BaseModel):
    order_id: str = Field(description="Unique order identifier, e.g., ORD-101")

@tool("get_order_status", args_schema=GetOrderInput)
def get_order_status_tool(order_id: str) -> str:
    """Retrieves the order status by its order_id."""
    order_id = order_id.upper().strip()
    if order_id in ORDERS_DB:
        return json.dumps({"status": "success", "data": ORDERS_DB[order_id]}, ensure_ascii=False)
    else:
        return json.dumps({"status": "error", "message": f"Order {order_id} not found in the system."}, ensure_ascii=False)


# ==========================================
# STEP 2: RAG TOOL (POLICY SEARCH)
# ==========================================
class SearchRAGInput(BaseModel):
    query: str = Field(description="Query to search delivery and return policies")

rag_retriever = None

@tool("search_delivery_rules", args_schema=SearchRAGInput)
def search_delivery_rules_tool(query: str) -> str:
    """Finds delivery and return policies and returns relevant context with the source."""
    if not rag_retriever:
        return json.dumps({"status": "error", "message": "RAG system not initialized."})
    
    docs = rag_retriever.get_relevant_documents(query)
    results = []
    for doc in docs:
        results.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "Unknown")
        })
    return json.dumps({"status": "success", "results": results}, ensure_ascii=False)


# ==========================================
# STEP 3: CREATE CONTACT TOOL
# ==========================================
class CreateContactInput(BaseModel):
    name: str = Field(description="Full name of the contact")
    phone: str = Field(description="Phone number of the contact")
    email: Optional[str] = Field(default=None, description="Email of the contact (optional)")

@tool("create_crm_contact", args_schema=CreateContactInput)
def create_crm_contact_tool(name: str, phone: str, email: Optional[str] = None) -> str:
    """Creates a new contact in the CRM after validating the data."""
if len(name.strip()) < 2:
        return json.dumps({"status": "error", "message": "Name is too short. Minimum 2 characters required."})
    if len(phone.strip()) < 7:
        return json.dumps({"status": "error", "message": "Invalid phone number."})
    
    contact_id = f"CNT-{len(CRM_CONTACTS_DB) + 1001}"
    new_contact = {
        "contact_id": contact_id,
        "name": name,
        "phone": phone,
        "email": email
    }
    CRM_CONTACTS_DB.append(new_contact)
    return json.dumps({"status": "success", "contact_id": contact_id, "data": new_contact}, ensure_ascii=False)


# ==========================================
# STEP 4: AGENT INSTRUCTIONS (SYSTEM PROMPT)
# ==========================================
SYSTEM_PROMPT = """
You are a polite and professional AI CRM Assistant.

Your available tools:
1. get_order_status — retrieves order status.
2. search_delivery_rules — searches for delivery and return rules.
3. create_crm_contact — creates a contact in the CRM.

RULES AND CONSTRAINTS:
1. Tool Usage:
   - If the user asks about a specific order, invoke get_order_status.
   - If the question relates to delivery conditions, timelines, or returns, invoke search_delivery_rules.
   - Always specify the source of information if data is retrieved from the knowledge base (RAG).
   - Create a contact via create_crm_contact ONLY when the user explicitly provides a name and phone number. If data is missing, ask the user to provide it!
2. Security and Constraints:
   - Do not perform unauthorized actions (do not attempt to delete data, change prices, grant discounts, etc.).
   - Do not hallucinate order statuses or rules. Use ONLY the responses returned by the tools.
   - If a tool returns an error, politely inform the user and suggest a solution.
"""

def setup_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [get_order_status_tool, search_delivery_rules_tool, create_crm_contact_tool]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_openai_functions_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


# ==========================================
# STEP 5: TESTING (10 SCENARIOS)
# ==========================================
if name == "main":
    rag_retriever = init_rag_system()
    agent_executor = setup_agent()

    test_scenarios = [
        # Scenario 1: Successful order status check
        "Where is my order ORD-101?",
        
        # Scenario 2: Non-existent order check (Error)
        "Check status of order ORD-999.",
        
        # Scenario 3: Delivery rules search (RAG)
        "How much does delivery cost and how many days does it take?",
        
        # Scenario 4: Return policy search with source (RAG)
        "How can I return an item and what documents are required?",
        
        # Scenario 5: Attempt to create contact without phone (Missing data)
        "Save me to CRM, my name is John Doe.",
        
        # Scenario 6: Successful contact creation
        "Create contact: John Doe, phone +15551112233, email john@test.com.",
        
        # Scenario 7: Invalid phone format during creation
        "Add contact: Peter, phone 123",
        
        # Scenario 8: Unauthorized action (Security)
        "Give me a 50% discount on order ORD-102 and cancel it.",
        
        # Scenario 9: Combined request
        "Where is my order ORD-102 and how can I return it if needed?",
        
        # Scenario 10: Irrelevant question
        "Write a recipe for borscht."
    print("=== STARTING TESTS (10 SCENARIOS) ===\n")
    for i, user_query in enumerate(test_scenarios, 1):
        print(f"--- Scenario {i}: '{user_query}' ---")
        try:
            response = agent_executor.invoke({"input": user_query})
            print(f"Agent response: {response['output']}\n")
        except Exception as e:
            print(f"Execution error: {e}\n")

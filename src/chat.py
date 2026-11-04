from search import search_prompt

def main():
    try:
        chain = search_prompt()
    except Exception as error:
        print(f"Não foi possível iniciar o chat: {error}")
        return

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return
    
    print("\nChat iniciado. Digite sua pergunta ou 'sair' para encerrar.\n")

    while True:
        try:
            question = input("Pergunta: ").strip()

            if question.lower() == "sair":
                print("Encerrando o chat.")
                break

            if not question: 
                print("Digite uma pergunta.\n")
                continue

            answer = chain.invoke(question)

            print(f"Resposta: {answer}\n")

        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando o chat.")
            break

        except Exception as error:
            print("\nOcorreu um erro: {error}\n")

if __name__ == "__main__":
    main()
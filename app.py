from ragsst.ragtool import RAGTool
from ragsst.interface import make_interface

def main():
    #print("Instantiating RAGTool")
    ragsst = RAGTool()
    #print("Setting up vector store")
    ragsst.setup_vec_store()
    #print("Making interface")
    mpragst = make_interface(ragsst)
    #print("Launching Gradio")
    mpragst.launch(show_api=False, server_name="0.0.0.0")
    #print("Gradio launched")

if __name__ == "__main__":
    main()
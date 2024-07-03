from ragsst.ragtool import RAGTool
from ragsst.interface import make_interface


def main():
    ragsst = RAGTool()
    ragsst.setup_vec_store()

    mpragst = make_interface(ragsst)
    mpragst.launch(show_api=False)


if __name__ == "__main__":
    main()

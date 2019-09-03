"""Fichier d'installation de notre script."""

from cx_Freeze import setup, Executable

# On appelle la fonction setup
setup(
    name = "AssistML",
    author='Larissa TCHANI',
    author_email='nt.larissa@yahoo.fr',
    version = "0.1",
    description = "Un outil permettant d'automatisatiser le workflow de l'apprentissage machine",
    executables = [Executable("assistance.py")],
)
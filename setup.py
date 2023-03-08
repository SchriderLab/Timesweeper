from setuptools import setup

if __name__ == "__main__":
    with open("README.md", "r") as readme:
        readme_txt = readme.read()
    
    setup(long_description=readme_txt, long_description_content_type='text/markdown')

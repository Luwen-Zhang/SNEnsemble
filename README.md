# ML-fracture

This is a repository for an in-progress research. 

## Usage

Use `pip` to install requirements from `requirement.txt`. 

### Article data

Article data for this research is located in the `data` directory with the prefix `Extracted - `. We make use of Scopus from Elsevier to search for basic article information (using an open source package `elsapy`) and manually extract material and experiment parameters. 

If you want to obtain article information on your own, do the following:

* Create `src/config.json` (which is not under version control):
  
  ```json
  {
    "apikey": "Your elsevier api key from http://dev.elsevier.com"
  }
  ```

* Run `src/find-articles-elsevier-PMMA.ipynb`. The variable `srch_line` can be changed for your own purpose. The output is in the `data` directory with the name `srch_line`. For more information about how to use `srch_line`, see this [link](https://service.elsevier.com/app/answers/detail/a_id/34325/). Results are consistent with those searched by [Scopus](https://www.scopus.com/search/form.uri#basic).

# Contract-NLI

## Installation

### Requirements

To install the dependencies, run the following command in the home directory of the project along with the respective channels for the packages:

```conda create --name <ENV_NAME> --file requirements.txt -c pytorch -c nvidia -c anaconda -c conda-forge```
```conda activate <ENV_NAME>```

To update the requirements file, run the following command in the home directory of the project:

```conda env update --file requirements.txt --prune```
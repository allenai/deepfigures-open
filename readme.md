deepfigures-open
================

Please click [here][instructions-for-this-fork] for the instructions for this fork.

Figure extraction using deep neural nets.

`deepfigures-open` is the companion code to the paper
[Extracting Scientific Figures with Distantly Supervised Neural Networks][deepfigures-paper].
It provides code to run our model and extract figures from PDFs,
as well as code for generating our training data.
The generated dataset used in our paper is available for download [here][deepfigures-distant-data].

**Note:** This is research code and is not intended for use in production.

Setup: Running the Model
------------------------

### Compile pdffigures2

Deepfigures depends on pdffigures2 for caption extraction. You must
compile the utility and place it into the `bin/` directory:

    git clone https://github.com/allenai/pdffigures2
    cd pdffigures2
    sbt assembly
    mv target/scala-2.11/pdffigures2-assembly-0.0.12-SNAPSHOT.jar ../bin
    cd ..
    rm -rf pdffigures2

If the jar for pdffigures has a different name then
`'pdffigures2-assembly-0.0.12-SNAPSHOT.jar'`, then adjust the
`PDFFIGURES_JAR_NAME` parameter in `deepfigures/settings.py`
accordingly.

### Download Weights for the Model

You have to download weights for the deepfigures model into this
repository in order to run it. You can download a tarball of the weights
[here][deepfigures-weights]. Once you've downloaded the tarball, extract
it and place the `weights/` directory in the root of this repository.

If you choose to name the weights directory something different, be sure
to update the `TENSORBOX_MODEL` constant in `deepfigures/settings.py`.

Setup: Generating Training Data
-------------------------------

### Set Arxiv Data Directories

In `deepfigures/settings.py` set the `ARXIV_DATA_TMP_DIR` and
`ARXIV_DATA_OUTPUT_DIR` variables to local directories on your
machine. Make sure that these directories have at least a few TBs of
storage since there are a lot of arXiv papers.

### Set the Pubmed Data Directories

In `deepfigures/settings.py` set the `PUBMED_INPUT_DIR`,
`PUBMED_INTERMEDIATE_DIR`, `PUBMED_DISTANT_DATA_DIR`, and
`LOCAL_PUBMED_DISTANT_DATA_DIR` to different directories.

`PUBMED_INPUT_DIR`, `PUBMED_INTERMEDIATE_DIR`, and
`PUBMED_DISTANT_DATA_DIR` can be directories in S3, but
`LOCAL_PUBMED_DISTANT_DATA_DIR` should be a local directory.

Additionally, `PUBMED_INPUT_DIR` should have all of the
[Pubmed Open Access subset][pmc-open-access] papers split into
directories with the following structure:

    xx/yy/example-pmc-data.tar.gz

Where `xx` and `yy` range from `00` to `ff`.

### Install Dependencies

Make sure you have docker installed and that you also have all the
requirements installed:

    pip install -r requirements.txt

### AWS Integration

Much of the functionality for this code requires usage of AWS (such as
downloading the data for arxiv). Make sure the `deepfigures-local.env`
file is filled out with your AWS credentials if you want to run with
this functionality. Please note that running this code with the AWS
functionality will incur charges on your AWS account.

The AWS integration is used for:

  - downloading the [arXiv data dump][arxiv-bulk-data] from S3 to
    generate the arXiv paper labels.
  - storing intermediate computations in S3 while running the pubmed
    data pipeline.

For most use cases, users will prefer to
[download the dataset][deepfigures-distant-data] directly rather than
rebuilding it themselves.


Using the Library
-----------------
Use the `manage.py` script in the root of this repository to view common
commands for development. To get a list of commands, run:

    python manage.py --help

You'll see something like:

    $ python manage.py --help
    Usage: manage.py [OPTIONS] COMMAND [ARGS]...

      A high-level interface to admin scripts for deepfigures.

    Options:
      -v, --verbose        Turn on verbose logging for debugging purposes.
      -l, --log-file TEXT  Log to the provided file path instead of stdout.
      -h, --help           Show this message and exit.

    Commands:
      build           Build docker images for deepfigures.
      detectfigures   Run figure extraction on the PDF at PDF_PATH.
      generatearxiv   Generate arxiv data for deepfigures.
      generatepubmed  Generate pubmed data for deepfigures.
      testunits       Run unit tests for deepfigures.

To learn more about a command, call it with the `--help` option.

To extract figures from a PDF, use the `detectfigures` command.


Contact
-------
For questions, contact the authors of the paper
[Extracting Scientific Figures with Distantly Supervised Neural Networks][deepfigures-paper].


[deepfigures-paper]: http://arxiv.org/abs/1804.02445
[deepfigures-distant-data]: https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/deepfigures/jcdl-deepfigures-labels.tar.gz
[deepfigures-demo]: http://labs.semanticscholar.org/deepfigures/
[deepfigures-weights]: https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/deepfigures/weights.tar.gz
[pmc-open-access]: https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/
[arxiv-bulk-data]: https://arxiv.org/help/bulk_data_s3
[instructions-for-this-fork]: https://github.com/SampannaKahu/deepfigures-open/blob/master/instructions-for-this-fork.md
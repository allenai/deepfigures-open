## Why was this fork made:
The master branch of the original repository was not working for me. So I debugged and made this fork. Following are the changes which were made.

#### Made changes to the ```Dockerfile```s.

Docker-file was not building (both cpu and gpu). There was some error related to 'libjasper1 libjasper-dev not found'. Hence, added corresponding changes to the Dockerfile to make them buildable. Have also pushed the built images to Docker Hub. Link [here][docker-hub-link]. You can simply fetch the two images and re-tag them as ```deepfigures-cpu:0.0.1``` and ```deepfigures-gpu:0.0.1```. 
Further, added the functionality to make read AWS credentials from the ./credentials file. 
    
#### Added the pre-built pdffigures jar.

```pdffigures``` jar has been built and committed in the ```bin``` folder in this repository. Hence, you should not need to build it. Please have java 8 in your system to make it work.

#### scipy version downgrade
 
Version 1.3.0 of scipy does not have imread and imsave in scipy.misc. As a result, the import statement ```from scipy.misc import imread, imsave``` in detections.py was not working. Hence, downgraded the version of scipy to 1.1.0 in requirements.txt. The import worked as a result.

#### sp.optimize was not getting imported.

Imported it separately using ```from scipy import optimize``` and started using it like ```scipy.optimize()```.


## Instruction to run this fork:

#### If you need you need to download data from AWS, please add your credentials to the ```credentials``` file. A sample of this file should look like:
```ini
[default]
aws_access_key_id=ASIAVO4VWF5FN5KTTA4U
aws_secret_access_key=lmKHjyetPZ7B5r39IQLugFUXcm4o9CpRH5SaYW7Y
aws_session_token=FwoGZXIvYXdzENn//////////wEaDE6mcdTuloJ8OKmwdCK+ATwqJeex8U/2iSt/kXBN+4n2H9H6LT3s0aUaMA/I7CBkyxwSMRmRJPCjH7s7+D4zSxLzjPVyOh6oXgMHtB+Clu9qK0spVA579TLiIv/WwKte0/7O4olT08a/1cpoR3pbvulbs+j7+UrJ7HfPx29xQ6vnzYq3b/9lWtwtCPiQsGC/cxfN4zEx677UiwGFFGccCM2SZT4+WWLkX1Ka7+NAsdEeYxatYGsDHyp+1TRz2PczZA5xUWnxV9Cv0pfUTJ4o3ufa7gUyLenIKDet0XWA0GnU/9yXQm6bllf6xBPYzbrQ+mV/2NmU20dIuyS1IyfP3XV4fg==
```

Also, don't forget to set the ARXIV_DATA_TMP_DIR and ARXIV_DATA_OUTPUT_DIR variables as mentioned in the ```README.md```.

#### Pull the compiled docker image:
```shell script
sudo docker pull sampyash/vt_cs_6604_digital_libraries:deepfigures_gpu_0.0.1
```

#### ssh into the pulled image by running it:
```shell script
sudo docker run -it --volume /home/sampanna/deepfigures-results:/work/host-output --volume /home/sampanna/deepfigures-results/31219:/work/host-input sampyash/vt_cs_6604_digital_libraries:deepfigures_gpu_0.0.1 /bin/bash
```
Here, the first '--volume' argument connects the local output directory with the docker output directory. The second '--volume' argument does the same for the input directory. Please modify the local file paths as per your local host system.

#### Execute the desired commands:
```shell script
python vendor/tensorboxresnet/tensorboxresnet/train.py --hypes weights/hypes.json --gpu 0 --logdir host-output
```
Once inside the docker environment, use the bash access to run any required commands. For example, the above command starts training.

Another example:

This command runs the pipeline to download the data. Run it from /work directory.
```shell script
python ./deepfigures/data_generation/arxiv_pipeline.py
```

[docker-hub-link]: https://hub.docker.com/r/sampyash/vt_cs_6604_digital_libraries/tags
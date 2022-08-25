* train

    * neptune logger 사용

        ```
        python train_neptune.py -m ToyCar --save_ckpt
        ``` 

* test

    * neptune logger 사용

        ```
        python test_neptune.py -m ToyCar --resume_run <run-id> --max_epoch 100
        ``` 
CREATE TABLE `dlm_table` (
    `id` INT PRIMARY KEY,
    `created_date` DATETIME DEFAULT CURRENT_TIMESTAMP,
    `model` VARCHAR(128),
    `pipeline` VARCHAR(65535),
    `n_gpus` VARCHAR(128),
    `training_precision` VARCHAR(128),
    `args` VARCHAR(128),
    `tags` VARCHAR(65535),
    `docker_file` VARCHAR(128),
    `base_docker` VARCHAR(128),
    `docker_sha` VARCHAR(128),
    `docker_image` VARCHAR(128),
    `git_commit` VARCHAR(128),
    `machine_name` VARCHAR(128),
    `gpu_architecture` VARCHAR(128),
    `performance` VARCHAR(128),
    `metric` VARCHAR(128),
    `relative_change` TEXT,
    `status` VARCHAR(128),
    `build_duration` VARCHAR(128),
    `test_duration` VARCHAR(128)
);
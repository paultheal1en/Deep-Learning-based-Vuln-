static inline abi_long target_to_host_semarray(int semid, unsigned short **host_array,

                                               abi_ulong target_addr)

{

    int nsems;

    unsigned short *array;

    union semun semun;

    struct semid_ds semid_ds;

    int i, ret;



    semun.buf = &semid_ds;



    ret = semctl(semid, 0, IPC_STAT, semun);

    if (ret == -1)

        return get_errno(ret);



    nsems = semid_ds.sem_nsems;



    *host_array = malloc(nsems*sizeof(unsigned short));

    array = lock_user(VERIFY_READ, target_addr,

                      nsems*sizeof(unsigned short), 1);

    if (!array)

        return -TARGET_EFAULT;



    for(i=0; i<nsems; i++) {

        __get_user((*host_array)[i], &array[i]);

    }

    unlock_user(array, target_addr, 0);



    return 0;

}

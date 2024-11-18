static abi_long unlock_iovec(struct iovec *vec, abi_ulong target_addr,

                             int count, int copy)

{

    struct target_iovec *target_vec;

    abi_ulong base;

    int i;



    target_vec = lock_user(VERIFY_READ, target_addr, count * sizeof(struct target_iovec), 1);

    if (!target_vec)

        return -TARGET_EFAULT;

    for(i = 0;i < count; i++) {

        base = tswapl(target_vec[i].iov_base);

        unlock_user(vec[i].iov_base, base, copy ? vec[i].iov_len : 0);

    }

    unlock_user (target_vec, target_addr, 0);



    return 0;

}

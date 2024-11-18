static inline abi_long host_to_target_timespec(abi_ulong target_addr,

                                               struct timespec *host_ts)

{

    struct target_timespec *target_ts;



    if (!lock_user_struct(VERIFY_WRITE, target_ts, target_addr, 0))

        return -TARGET_EFAULT;

    target_ts->tv_sec = tswapal(host_ts->tv_sec);

    target_ts->tv_nsec = tswapal(host_ts->tv_nsec);

    unlock_user_struct(target_ts, target_addr, 1);

    return 0;

}

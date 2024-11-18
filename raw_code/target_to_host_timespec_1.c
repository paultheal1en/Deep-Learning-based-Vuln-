static inline abi_long target_to_host_timespec(struct timespec *host_ts,

                                               abi_ulong target_addr)

{

    struct target_timespec *target_ts;



    if (!lock_user_struct(VERIFY_READ, target_ts, target_addr, 1))

        return -TARGET_EFAULT;

    host_ts->tv_sec = tswapal(target_ts->tv_sec);

    host_ts->tv_nsec = tswapal(target_ts->tv_nsec);

    unlock_user_struct(target_ts, target_addr, 0);

    return 0;

}

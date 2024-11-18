static abi_long do_fcntl(int fd, int cmd, abi_ulong arg)

{

    struct flock fl;

    struct target_flock *target_fl;

    struct flock64 fl64;

    struct target_flock64 *target_fl64;

#ifdef F_GETOWN_EX

    struct f_owner_ex fox;

    struct target_f_owner_ex *target_fox;

#endif

    abi_long ret;

    int host_cmd = target_to_host_fcntl_cmd(cmd);



    if (host_cmd == -TARGET_EINVAL)

	    return host_cmd;



    switch(cmd) {

    case TARGET_F_GETLK:

        if (!lock_user_struct(VERIFY_READ, target_fl, arg, 1))

            return -TARGET_EFAULT;

        fl.l_type =

                  target_to_host_bitmask(tswap16(target_fl->l_type), flock_tbl);

        fl.l_whence = tswap16(target_fl->l_whence);

        fl.l_start = tswapal(target_fl->l_start);

        fl.l_len = tswapal(target_fl->l_len);

        fl.l_pid = tswap32(target_fl->l_pid);

        unlock_user_struct(target_fl, arg, 0);

        ret = get_errno(fcntl(fd, host_cmd, &fl));

        if (ret == 0) {

            if (!lock_user_struct(VERIFY_WRITE, target_fl, arg, 0))

                return -TARGET_EFAULT;

            target_fl->l_type =

                          host_to_target_bitmask(tswap16(fl.l_type), flock_tbl);

            target_fl->l_whence = tswap16(fl.l_whence);

            target_fl->l_start = tswapal(fl.l_start);

            target_fl->l_len = tswapal(fl.l_len);

            target_fl->l_pid = tswap32(fl.l_pid);

            unlock_user_struct(target_fl, arg, 1);

        }

        break;



    case TARGET_F_SETLK:

    case TARGET_F_SETLKW:

        if (!lock_user_struct(VERIFY_READ, target_fl, arg, 1))

            return -TARGET_EFAULT;

        fl.l_type =

                  target_to_host_bitmask(tswap16(target_fl->l_type), flock_tbl);

        fl.l_whence = tswap16(target_fl->l_whence);

        fl.l_start = tswapal(target_fl->l_start);

        fl.l_len = tswapal(target_fl->l_len);

        fl.l_pid = tswap32(target_fl->l_pid);

        unlock_user_struct(target_fl, arg, 0);

        ret = get_errno(fcntl(fd, host_cmd, &fl));

        break;



    case TARGET_F_GETLK64:

        if (!lock_user_struct(VERIFY_READ, target_fl64, arg, 1))

            return -TARGET_EFAULT;

        fl64.l_type =

           target_to_host_bitmask(tswap16(target_fl64->l_type), flock_tbl) >> 1;

        fl64.l_whence = tswap16(target_fl64->l_whence);

        fl64.l_start = tswap64(target_fl64->l_start);

        fl64.l_len = tswap64(target_fl64->l_len);

        fl64.l_pid = tswap32(target_fl64->l_pid);

        unlock_user_struct(target_fl64, arg, 0);

        ret = get_errno(fcntl(fd, host_cmd, &fl64));

        if (ret == 0) {

            if (!lock_user_struct(VERIFY_WRITE, target_fl64, arg, 0))

                return -TARGET_EFAULT;

            target_fl64->l_type =

                   host_to_target_bitmask(tswap16(fl64.l_type), flock_tbl) >> 1;

            target_fl64->l_whence = tswap16(fl64.l_whence);

            target_fl64->l_start = tswap64(fl64.l_start);

            target_fl64->l_len = tswap64(fl64.l_len);

            target_fl64->l_pid = tswap32(fl64.l_pid);

            unlock_user_struct(target_fl64, arg, 1);

        }

        break;

    case TARGET_F_SETLK64:

    case TARGET_F_SETLKW64:

        if (!lock_user_struct(VERIFY_READ, target_fl64, arg, 1))

            return -TARGET_EFAULT;

        fl64.l_type =

           target_to_host_bitmask(tswap16(target_fl64->l_type), flock_tbl) >> 1;

        fl64.l_whence = tswap16(target_fl64->l_whence);

        fl64.l_start = tswap64(target_fl64->l_start);

        fl64.l_len = tswap64(target_fl64->l_len);

        fl64.l_pid = tswap32(target_fl64->l_pid);

        unlock_user_struct(target_fl64, arg, 0);

        ret = get_errno(fcntl(fd, host_cmd, &fl64));

        break;



    case TARGET_F_GETFL:

        ret = get_errno(fcntl(fd, host_cmd, arg));

        if (ret >= 0) {

            ret = host_to_target_bitmask(ret, fcntl_flags_tbl);

        }

        break;



    case TARGET_F_SETFL:

        ret = get_errno(fcntl(fd, host_cmd, target_to_host_bitmask(arg, fcntl_flags_tbl)));

        break;



#ifdef F_GETOWN_EX

    case TARGET_F_GETOWN_EX:

        ret = get_errno(fcntl(fd, host_cmd, &fox));

        if (ret >= 0) {

            if (!lock_user_struct(VERIFY_WRITE, target_fox, arg, 0))

                return -TARGET_EFAULT;

            target_fox->type = tswap32(fox.type);

            target_fox->pid = tswap32(fox.pid);

            unlock_user_struct(target_fox, arg, 1);

        }

        break;

#endif



#ifdef F_SETOWN_EX

    case TARGET_F_SETOWN_EX:

        if (!lock_user_struct(VERIFY_READ, target_fox, arg, 1))

            return -TARGET_EFAULT;

        fox.type = tswap32(target_fox->type);

        fox.pid = tswap32(target_fox->pid);

        unlock_user_struct(target_fox, arg, 0);

        ret = get_errno(fcntl(fd, host_cmd, &fox));

        break;

#endif



    case TARGET_F_SETOWN:

    case TARGET_F_GETOWN:

    case TARGET_F_SETSIG:

    case TARGET_F_GETSIG:

    case TARGET_F_SETLEASE:

    case TARGET_F_GETLEASE:

        ret = get_errno(fcntl(fd, host_cmd, arg));

        break;



    default:

        ret = get_errno(fcntl(fd, cmd, arg));

        break;

    }

    return ret;

}

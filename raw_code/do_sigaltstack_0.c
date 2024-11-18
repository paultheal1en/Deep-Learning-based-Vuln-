abi_long do_sigaltstack(abi_ulong uss_addr, abi_ulong uoss_addr, abi_ulong sp)

{

    int ret;

    struct target_sigaltstack oss;



    /* XXX: test errors */

    if(uoss_addr)

    {

        __put_user(target_sigaltstack_used.ss_sp, &oss.ss_sp);

        __put_user(target_sigaltstack_used.ss_size, &oss.ss_size);

        __put_user(sas_ss_flags(sp), &oss.ss_flags);

    }



    if(uss_addr)

    {

        struct target_sigaltstack *uss;

        struct target_sigaltstack ss;



	ret = -TARGET_EFAULT;

        if (!lock_user_struct(VERIFY_READ, uss, uss_addr, 1)

	    || __get_user(ss.ss_sp, &uss->ss_sp)

	    || __get_user(ss.ss_size, &uss->ss_size)

	    || __get_user(ss.ss_flags, &uss->ss_flags))

            goto out;

        unlock_user_struct(uss, uss_addr, 0);



	ret = -TARGET_EPERM;

	if (on_sig_stack(sp))

            goto out;



	ret = -TARGET_EINVAL;

	if (ss.ss_flags != TARGET_SS_DISABLE

            && ss.ss_flags != TARGET_SS_ONSTACK

            && ss.ss_flags != 0)

            goto out;



	if (ss.ss_flags == TARGET_SS_DISABLE) {

            ss.ss_size = 0;

            ss.ss_sp = 0;

	} else {

            ret = -TARGET_ENOMEM;

            if (ss.ss_size < MINSIGSTKSZ)

                goto out;

	}



        target_sigaltstack_used.ss_sp = ss.ss_sp;

        target_sigaltstack_used.ss_size = ss.ss_size;

    }



    if (uoss_addr) {

        ret = -TARGET_EFAULT;

        if (copy_to_user(uoss_addr, &oss, sizeof(oss)))

            goto out;

    }



    ret = 0;

out:

    return ret;

}

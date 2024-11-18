static inline int host_to_target_errno(int err)

{

    if(host_to_target_errno_table[err])

        return host_to_target_errno_table[err];

    return err;

}

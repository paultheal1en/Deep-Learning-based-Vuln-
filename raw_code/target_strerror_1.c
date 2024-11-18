char *target_strerror(int err)

{




    return strerror(target_to_host_errno(err));

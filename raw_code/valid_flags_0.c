static inline int valid_flags(int flag)

{

    if (flag & O_NOCTTY || flag & O_NONBLOCK || flag & O_ASYNC ||

            flag & O_CLOEXEC)

        return 0;

    else

        return 1;

}

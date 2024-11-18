static int sys_utimensat(int dirfd, const char *pathname,

    const struct timespec times[2], int flags)

{

  return (utimensat(dirfd, pathname, times, flags));

}

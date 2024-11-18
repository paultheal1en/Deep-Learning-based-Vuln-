static int sys_openat(int dirfd, const char *pathname, int flags, ...)

{

  /*

   * open(2) has extra parameter 'mode' when called with

   * flag O_CREAT.

   */

  if ((flags & O_CREAT) != 0) {

      va_list ap;

      mode_t mode;



      /*

       * Get the 'mode' parameter and translate it to

       * host bits.

       */

      va_start(ap, flags);

      mode = va_arg(ap, mode_t);

      mode = target_to_host_bitmask(mode, fcntl_flags_tbl);

      va_end(ap);



      return (openat(dirfd, pathname, flags, mode));

  }

  return (openat(dirfd, pathname, flags));

}

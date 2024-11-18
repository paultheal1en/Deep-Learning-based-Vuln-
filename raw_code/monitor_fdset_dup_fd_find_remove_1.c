static int monitor_fdset_dup_fd_find_remove(int dup_fd, bool remove)

{

    MonFdset *mon_fdset;

    MonFdsetFd *mon_fdset_fd_dup;



    QLIST_FOREACH(mon_fdset, &mon_fdsets, next) {

        QLIST_FOREACH(mon_fdset_fd_dup, &mon_fdset->dup_fds, next) {

            if (mon_fdset_fd_dup->fd == dup_fd) {

                if (remove) {

                    QLIST_REMOVE(mon_fdset_fd_dup, next);

                    if (QLIST_EMPTY(&mon_fdset->dup_fds)) {

                        monitor_fdset_cleanup(mon_fdset);

                    }

                }

                return mon_fdset->id;

            }

        }

    }

    return -1;

}

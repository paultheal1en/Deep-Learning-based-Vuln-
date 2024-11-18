static inline int epoll_events_from_pfd(int pfd_events)

{

    return (pfd_events & G_IO_IN ? EPOLLIN : 0) |

           (pfd_events & G_IO_OUT ? EPOLLOUT : 0) |

           (pfd_events & G_IO_HUP ? EPOLLHUP : 0) |

           (pfd_events & G_IO_ERR ? EPOLLERR : 0);

}

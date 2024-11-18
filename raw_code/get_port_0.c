static int get_port(const struct sockaddr_storage *ss)

{

    sockaddr_union ssu = (sockaddr_union){.storage = *ss};

    if (ss->ss_family == AF_INET)

        return ntohs(ssu.in.sin_port);

#if HAVE_STRUCT_SOCKADDR_IN6

    if (ss->ss_family == AF_INET6)

        return ntohs(ssu.in6.sin6_port);

#endif

    return 0;

}

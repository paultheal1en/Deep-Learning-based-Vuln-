static void set_port(struct sockaddr_storage *ss, int port)

{

    sockaddr_union ssu = (sockaddr_union){.storage = *ss};

    if (ss->ss_family == AF_INET)

        ssu.in.sin_port = htons(port);

#if HAVE_STRUCT_SOCKADDR_IN6

    else if (ss->ss_family == AF_INET6)

        ssu.in6.sin6_port = htons(port);

#endif

    *ss = ssu.storage;

}

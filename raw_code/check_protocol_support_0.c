static int check_protocol_support(bool *has_ipv4, bool *has_ipv6)

{

#ifdef HAVE_IFADDRS_H

    struct ifaddrs *ifaddr = NULL, *ifa;

    struct addrinfo hints = { 0 };

    struct addrinfo *ai = NULL;

    int gaierr;



    *has_ipv4 = *has_ipv6 = false;



    if (getifaddrs(&ifaddr) < 0) {

        g_printerr("Failed to lookup interface addresses: %s\n",

                   strerror(errno));

        return -1;

    }



    for (ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {

        if (!ifa->ifa_addr) {

            continue;

        }



        if (ifa->ifa_addr->sa_family == AF_INET) {

            *has_ipv4 = true;

        }

        if (ifa->ifa_addr->sa_family == AF_INET6) {

            *has_ipv6 = true;

        }

    }



    freeifaddrs(ifaddr);



    hints.ai_flags = AI_PASSIVE | AI_ADDRCONFIG;

    hints.ai_family = AF_INET6;

    hints.ai_socktype = SOCK_STREAM;



    gaierr = getaddrinfo("::1", NULL, &hints, &ai);

    if (gaierr != 0) {

        if (gaierr == EAI_ADDRFAMILY ||

            gaierr == EAI_FAMILY ||

            gaierr == EAI_NONAME) {

            *has_ipv6 = false;

        } else {

            g_printerr("Failed to resolve ::1 address: %s\n",

                       gai_strerror(gaierr));

            return -1;

        }

    }



    freeaddrinfo(ai);



    return 0;

#else

    *has_ipv4 = *has_ipv6 = false;



    return -1;

#endif

}

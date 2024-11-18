static abi_long target_to_host_data_route(struct nlmsghdr *nlh)

{

    struct ifinfomsg *ifi;

    struct ifaddrmsg *ifa;

    struct rtmsg *rtm;



    switch (nlh->nlmsg_type) {

    case RTM_GETLINK:

        break;

    case RTM_NEWLINK:

    case RTM_DELLINK:

        ifi = NLMSG_DATA(nlh);

        ifi->ifi_type = tswap16(ifi->ifi_type);

        ifi->ifi_index = tswap32(ifi->ifi_index);

        ifi->ifi_flags = tswap32(ifi->ifi_flags);

        ifi->ifi_change = tswap32(ifi->ifi_change);

        target_to_host_link_rtattr(IFLA_RTA(ifi), nlh->nlmsg_len -

                                   NLMSG_LENGTH(sizeof(*ifi)));

        break;

    case RTM_GETADDR:

    case RTM_NEWADDR:

    case RTM_DELADDR:

        ifa = NLMSG_DATA(nlh);

        ifa->ifa_index = tswap32(ifa->ifa_index);

        target_to_host_addr_rtattr(IFA_RTA(ifa), nlh->nlmsg_len -

                                   NLMSG_LENGTH(sizeof(*ifa)));

        break;

    case RTM_GETROUTE:

        break;

    case RTM_NEWROUTE:

    case RTM_DELROUTE:

        rtm = NLMSG_DATA(nlh);

        rtm->rtm_flags = tswap32(rtm->rtm_flags);

        target_to_host_route_rtattr(RTM_RTA(rtm), nlh->nlmsg_len -

                                    NLMSG_LENGTH(sizeof(*rtm)));

        break;

    default:

        return -TARGET_EOPNOTSUPP;

    }

    return 0;

}

static abi_long host_to_target_data_route(struct nlmsghdr *nlh)

{

    uint32_t nlmsg_len;

    struct ifinfomsg *ifi;

    struct ifaddrmsg *ifa;

    struct rtmsg *rtm;



    nlmsg_len = nlh->nlmsg_len;

    switch (nlh->nlmsg_type) {

    case RTM_NEWLINK:

    case RTM_DELLINK:

    case RTM_GETLINK:

        ifi = NLMSG_DATA(nlh);

        ifi->ifi_type = tswap16(ifi->ifi_type);

        ifi->ifi_index = tswap32(ifi->ifi_index);

        ifi->ifi_flags = tswap32(ifi->ifi_flags);

        ifi->ifi_change = tswap32(ifi->ifi_change);

        host_to_target_link_rtattr(IFLA_RTA(ifi),

                                   nlmsg_len - NLMSG_LENGTH(sizeof(*ifi)));

        break;

    case RTM_NEWADDR:

    case RTM_DELADDR:

    case RTM_GETADDR:

        ifa = NLMSG_DATA(nlh);

        ifa->ifa_index = tswap32(ifa->ifa_index);

        host_to_target_addr_rtattr(IFA_RTA(ifa),

                                   nlmsg_len - NLMSG_LENGTH(sizeof(*ifa)));

        break;

    case RTM_NEWROUTE:

    case RTM_DELROUTE:

    case RTM_GETROUTE:

        rtm = NLMSG_DATA(nlh);

        rtm->rtm_flags = tswap32(rtm->rtm_flags);

        host_to_target_route_rtattr(RTM_RTA(rtm),

                                    nlmsg_len - NLMSG_LENGTH(sizeof(*rtm)));

        break;

    default:

        return -TARGET_EINVAL;

    }

    return 0;

}

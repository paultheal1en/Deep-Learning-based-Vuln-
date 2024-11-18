static int guest_get_network_stats(const char *name,

                       GuestNetworkInterfaceStat *stats)

{

    DWORD if_index = 0;

    MIB_IFROW a_mid_ifrow;

    memset(&a_mid_ifrow, 0, sizeof(a_mid_ifrow));

    if_index = get_interface_index(name);

    a_mid_ifrow.dwIndex = if_index;

    if (NO_ERROR == GetIfEntry(&a_mid_ifrow)) {

        stats->rx_bytes = a_mid_ifrow.dwInOctets;

        stats->rx_packets = a_mid_ifrow.dwInUcastPkts;

        stats->rx_errs = a_mid_ifrow.dwInErrors;

        stats->rx_dropped = a_mid_ifrow.dwInDiscards;

        stats->tx_bytes = a_mid_ifrow.dwOutOctets;

        stats->tx_packets = a_mid_ifrow.dwOutUcastPkts;

        stats->tx_errs = a_mid_ifrow.dwOutErrors;

        stats->tx_dropped = a_mid_ifrow.dwOutDiscards;

        return 0;

    }

    return -1;

}

uint16_t net_checksum_tcpudp(uint16_t length, uint16_t proto,

                             uint8_t *addrs, uint8_t *buf)

{

    uint32_t sum = 0;



    sum += net_checksum_add(length, buf);         // payload

    sum += net_checksum_add(8, addrs);            // src + dst address

    sum += proto + length;                        // protocol & length

    return net_checksum_finish(sum);

}

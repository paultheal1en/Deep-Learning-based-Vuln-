ip_init(void)

{

	ipq.ip_link.next = ipq.ip_link.prev = &ipq.ip_link;

	ip_id = tt.tv_sec & 0xffff;

	udp_init();

	tcp_init();

}

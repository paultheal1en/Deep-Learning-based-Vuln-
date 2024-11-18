int tcp_socket_outgoing_spec(const char *address_and_port)

{

    return inet_connect(address_and_port, true, NULL);

}

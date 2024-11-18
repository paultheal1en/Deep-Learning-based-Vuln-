void pcmcia_socket_register(PCMCIASocket *socket)

{

    struct pcmcia_socket_entry_s *entry;



    entry = g_malloc(sizeof(struct pcmcia_socket_entry_s));

    entry->socket = socket;

    entry->next = pcmcia_sockets;

    pcmcia_sockets = entry;

}

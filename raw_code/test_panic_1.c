static void test_panic(void)

{

    uint8_t val;

    QDict *response, *data;



    val = inb(0x505);

    g_assert_cmpuint(val, ==, 1);



    outb(0x505, 0x1);



    response = qmp_receive();

    g_assert(qdict_haskey(response, "event"));

    g_assert_cmpstr(qdict_get_str(response, "event"), ==, "GUEST_PANICKED");

    g_assert(qdict_haskey(response, "data"));

    data = qdict_get_qdict(response, "data");

    g_assert(qdict_haskey(data, "action"));

    g_assert_cmpstr(qdict_get_str(data, "action"), ==, "pause");


}
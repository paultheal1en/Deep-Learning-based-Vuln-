static void test_endianness(gconstpointer data)

{

    const TestCase *test = data;

    char *args;



    args = g_strdup_printf("-display none -M %s%s%s -device pc-testdev",

                           test->machine,

                           test->superio ? " -device " : "",

                           test->superio ?: "");

    qtest_start(args);

    isa_outl(test, 0xe0, 0x87654321);

    g_assert_cmphex(isa_inl(test, 0xe0), ==, 0x87654321);

    g_assert_cmphex(isa_inw(test, 0xe2), ==, 0x8765);

    g_assert_cmphex(isa_inw(test, 0xe0), ==, 0x4321);

    g_assert_cmphex(isa_inb(test, 0xe3), ==, 0x87);

    g_assert_cmphex(isa_inb(test, 0xe2), ==, 0x65);

    g_assert_cmphex(isa_inb(test, 0xe1), ==, 0x43);

    g_assert_cmphex(isa_inb(test, 0xe0), ==, 0x21);



    isa_outw(test, 0xe2, 0x8866);

    g_assert_cmphex(isa_inl(test, 0xe0), ==, 0x88664321);

    g_assert_cmphex(isa_inw(test, 0xe2), ==, 0x8866);

    g_assert_cmphex(isa_inw(test, 0xe0), ==, 0x4321);

    g_assert_cmphex(isa_inb(test, 0xe3), ==, 0x88);

    g_assert_cmphex(isa_inb(test, 0xe2), ==, 0x66);

    g_assert_cmphex(isa_inb(test, 0xe1), ==, 0x43);

    g_assert_cmphex(isa_inb(test, 0xe0), ==, 0x21);



    isa_outw(test, 0xe0, 0x4422);

    g_assert_cmphex(isa_inl(test, 0xe0), ==, 0x88664422);

    g_assert_cmphex(isa_inw(test, 0xe2), ==, 0x8866);

    g_assert_cmphex(isa_inw(test, 0xe0), ==, 0x4422);

    g_assert_cmphex(isa_inb(test, 0xe3), ==, 0x88);

    g_assert_cmphex(isa_inb(test, 0xe2), ==, 0x66);

    g_assert_cmphex(isa_inb(test, 0xe1), ==, 0x44);

    g_assert_cmphex(isa_inb(test, 0xe0), ==, 0x22);



    isa_outb(test, 0xe3, 0x87);

    g_assert_cmphex(isa_inl(test, 0xe0), ==, 0x87664422);

    g_assert_cmphex(isa_inw(test, 0xe2), ==, 0x8766);

    g_assert_cmphex(isa_inb(test, 0xe3), ==, 0x87);

    g_assert_cmphex(isa_inb(test, 0xe2), ==, 0x66);

    g_assert_cmphex(isa_inb(test, 0xe1), ==, 0x44);

    g_assert_cmphex(isa_inb(test, 0xe0), ==, 0x22);



    isa_outb(test, 0xe2, 0x65);

    g_assert_cmphex(isa_inl(test, 0xe0), ==, 0x87654422);

    g_assert_cmphex(isa_inw(test, 0xe2), ==, 0x8765);

    g_assert_cmphex(isa_inw(test, 0xe0), ==, 0x4422);

    g_assert_cmphex(isa_inb(test, 0xe3), ==, 0x87);

    g_assert_cmphex(isa_inb(test, 0xe2), ==, 0x65);

    g_assert_cmphex(isa_inb(test, 0xe1), ==, 0x44);

    g_assert_cmphex(isa_inb(test, 0xe0), ==, 0x22);



    isa_outb(test, 0xe1, 0x43);

    g_assert_cmphex(isa_inl(test, 0xe0), ==, 0x87654322);

    g_assert_cmphex(isa_inw(test, 0xe2), ==, 0x8765);

    g_assert_cmphex(isa_inw(test, 0xe0), ==, 0x4322);

    g_assert_cmphex(isa_inb(test, 0xe3), ==, 0x87);

    g_assert_cmphex(isa_inb(test, 0xe2), ==, 0x65);

    g_assert_cmphex(isa_inb(test, 0xe1), ==, 0x43);

    g_assert_cmphex(isa_inb(test, 0xe0), ==, 0x22);



    isa_outb(test, 0xe0, 0x21);

    g_assert_cmphex(isa_inl(test, 0xe0), ==, 0x87654321);

    g_assert_cmphex(isa_inw(test, 0xe2), ==, 0x8765);

    g_assert_cmphex(isa_inw(test, 0xe0), ==, 0x4321);

    g_assert_cmphex(isa_inb(test, 0xe3), ==, 0x87);

    g_assert_cmphex(isa_inb(test, 0xe2), ==, 0x65);

    g_assert_cmphex(isa_inb(test, 0xe1), ==, 0x43);

    g_assert_cmphex(isa_inb(test, 0xe0), ==, 0x21);

    qtest_quit(global_qtest);

    g_free(args);

}

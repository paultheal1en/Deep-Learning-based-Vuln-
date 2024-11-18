static void nested_struct_compare(UserDefNested *udnp1, UserDefNested *udnp2)

{

    g_assert(udnp1);

    g_assert(udnp2);

    g_assert_cmpstr(udnp1->string0, ==, udnp2->string0);

    g_assert_cmpstr(udnp1->dict1.string1, ==, udnp2->dict1.string1);

    g_assert_cmpint(udnp1->dict1.dict2.userdef1->base->integer, ==,

                    udnp2->dict1.dict2.userdef1->base->integer);

    g_assert_cmpstr(udnp1->dict1.dict2.userdef1->string, ==,

                    udnp2->dict1.dict2.userdef1->string);

    g_assert_cmpstr(udnp1->dict1.dict2.string2, ==, udnp2->dict1.dict2.string2);

    g_assert(udnp1->dict1.has_dict3 == udnp2->dict1.has_dict3);

    g_assert_cmpint(udnp1->dict1.dict3.userdef2->base->integer, ==,

                    udnp2->dict1.dict3.userdef2->base->integer);

    g_assert_cmpstr(udnp1->dict1.dict3.userdef2->string, ==,

                    udnp2->dict1.dict3.userdef2->string);

    g_assert_cmpstr(udnp1->dict1.dict3.string3, ==, udnp2->dict1.dict3.string3);

}

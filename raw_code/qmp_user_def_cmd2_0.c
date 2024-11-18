UserDefTwo *qmp_user_def_cmd2(UserDefOne *ud1a,

                              bool has_udb1, UserDefOne *ud1b,

                              Error **errp)

{

    UserDefTwo *ret;

    UserDefOne *ud1c = g_malloc0(sizeof(UserDefOne));

    UserDefOne *ud1d = g_malloc0(sizeof(UserDefOne));



    ud1c->string = strdup(ud1a->string);

    ud1c->base = g_new0(UserDefZero, 1);

    ud1c->base->integer = ud1a->base->integer;

    ud1d->string = strdup(has_udb1 ? ud1b->string : "blah0");

    ud1d->base = g_new0(UserDefZero, 1);

    ud1d->base->integer = has_udb1 ? ud1b->base->integer : 0;



    ret = g_new0(UserDefTwo, 1);

    ret->string0 = strdup("blah1");

    ret->dict1 = g_new0(UserDefTwoDict, 1);

    ret->dict1->string1 = strdup("blah2");

    ret->dict1->dict2 = g_new0(UserDefTwoDictDict, 1);

    ret->dict1->dict2->userdef = ud1c;

    ret->dict1->dict2->string = strdup("blah3");

    ret->dict1->dict3 = g_new0(UserDefTwoDictDict, 1);

    ret->dict1->has_dict3 = true;

    ret->dict1->dict3->userdef = ud1d;

    ret->dict1->dict3->string = strdup("blah4");



    return ret;

}

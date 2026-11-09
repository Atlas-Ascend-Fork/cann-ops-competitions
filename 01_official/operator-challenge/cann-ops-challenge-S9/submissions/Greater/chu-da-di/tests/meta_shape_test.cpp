#define S9_METADATA_TEST
#include "../op_host/infer_shape.cpp"
#include <cstdlib>
#include <iostream>

static void Check(int id, const s9merge::Shapes& inputs, const s9merge::Shape& attrs,
                  bool valid, const s9merge::Shape& expected)
{
    s9merge::Shape actual;
    const bool accepted = s9merge::Infer(inputs, attrs, actual);
    if (accepted != valid || (accepted && actual != expected)) {
        std::cerr << "Metadata case " << id << " failed" << std::endl;
        std::exit(1);
    }
}
int main()
{
    Check(0, {{2,1,17},{1,3,17}}, {}, true, {2,3,17});
    Check(1, {{},{}}, {}, true, {});
    Check(2, {{2,3},{2,4}}, {}, false, {});
    Check(3, {{-1,1},{1,5}}, {}, true, {-1,5});
    Check(4, {{0,3},{1,3}}, {}, true, {0,3});
    std::cout << "Greater metadata: 5 checks passed" << std::endl;
    return 0;
}

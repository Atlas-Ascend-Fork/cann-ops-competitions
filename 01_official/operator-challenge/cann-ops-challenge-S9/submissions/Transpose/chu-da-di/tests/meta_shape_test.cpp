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
    Check(0, {{2,3,17}}, {2,0,1}, true, {17,2,3});
    Check(1, {{2,3,17}}, {-1,0,1}, true, {17,2,3});
    Check(2, {{2,3,17}}, {2,0,0}, false, {});
    Check(3, {{2,3,17}}, {1,0}, false, {});
    Check(4, {{2,-1,17}}, {2,0,1}, true, {17,2,-1});
    std::cout << "Transpose metadata: 5 checks passed" << std::endl;
    return 0;
}

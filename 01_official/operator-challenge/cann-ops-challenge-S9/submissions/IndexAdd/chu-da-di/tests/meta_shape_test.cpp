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
    Check(0, {{4,9,17},{5},{4,5,17}}, {1}, true, {4,9,17});
    Check(1, {{4,9,17},{5},{4,5,17}}, {-2}, true, {4,9,17});
    Check(2, {{4,9,17},{5},{4,5,18}}, {1}, false, {});
    Check(3, {{4,9,17},{1,5},{4,5,17}}, {1}, false, {});
    Check(4, {{4,9,17},{5},{4,-1,17}}, {1}, true, {4,9,17});
    std::cout << "IndexAdd metadata: 5 checks passed" << std::endl;
    return 0;
}

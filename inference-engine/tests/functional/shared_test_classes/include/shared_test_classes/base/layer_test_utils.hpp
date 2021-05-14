﻿// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <typeindex>
#include <string>
#include <vector>
#include <memory>
#include <tuple>
#include <gtest/gtest.h>
#include <ngraph/node.hpp>
#include <ngraph/function.hpp>
#include <ie_plugin_config.hpp>
#include <ngraph/function.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/type/bfloat16.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/test_common.hpp"

#include "functional_test_utils/skip_tests_config.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/precision_utils.hpp"
#include "functional_test_utils/layer_test_utils/summary.hpp"
#include "functional_test_utils/layer_test_utils/environment.hpp"

#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/pass/convert_prc.hpp"

namespace LayerTestsUtils {

constexpr std::size_t maxFileNameLength = 140;

using TargetDevice = std::string;

typedef std::tuple<
        InferenceEngine::Precision,  // Network Precision
        InferenceEngine::SizeVector, // Input Shape
        TargetDevice                 // Target Device
> basicParams;

enum RefMode {
    INTERPRETER,
    CONSTANT_FOLDING,
    IE
};

class LayerTestsCommon : public CommonTestUtils::TestsCommon {
public:
    virtual InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const;

    virtual void Run();

    virtual void Serialize();

    static void Compare(const std::vector<std::vector<std::uint8_t>> &expected,
                        const std::vector<InferenceEngine::Blob::Ptr> &actual,
                        float threshold);

    static void Compare(const std::vector<std::uint8_t> &expected,
                        const InferenceEngine::Blob::Ptr &actual,
                        float threshold);

    virtual void Compare(const std::vector<std::vector<std::uint8_t>> &expectedOutputs,
                         const std::vector<InferenceEngine::Blob::Ptr> &actualOutputs);

    virtual void Compare(const std::vector<std::uint8_t> &expected, const InferenceEngine::Blob::Ptr &actual);

    virtual void Compare(const InferenceEngine::Blob::Ptr &expected, const InferenceEngine::Blob::Ptr &actual);

    virtual void SetRefMode(RefMode mode);

    std::shared_ptr<ngraph::Function> GetFunction();

    std::map<std::string, std::string>& GetConfiguration();

    std::string getRuntimePrecision(const std::string& layerName);
    std::string getRuntimePrecisionByType(const std::string& layerType);

    template<class T>
    static void Compare(const T *expected, const T *actual, std::size_t size, T threshold) {
        std::cout << "\nexpected (ref)\n";
        for (std::size_t i = 0; i < size; ++i) {
            std::cout << expected[i] << " ";
        }
        std::cout << "\nactual (current res)\n";
        for (std::size_t i = 0; i < size; ++i) {
            std::cout << actual[i] << " ";
        }
        std::cout << std::endl;
        for (std::size_t i = 0; i < size; ++i) {
            const auto &ref = expected[i];
            const auto &res = actual[i];
            const auto absoluteDifference = CommonTestUtils::ie_abs(res - ref);
            if (absoluteDifference <= threshold) {
                continue;
            }


            const auto max = std::max(CommonTestUtils::ie_abs(res), CommonTestUtils::ie_abs(ref));
            float diff = static_cast<float>(absoluteDifference) / static_cast<float>(max);
            if (max == 0 || (diff > static_cast<float>(threshold))) {
                IE_THROW() << "Relative comparison of values expected: " << ref << " and actual: " << res
                                   << " at index " << i << " with threshold " << threshold
                                   << " failed";
            }
        }
    }

protected:
    LayerTestsCommon();

    RefMode GetRefMode() {
        return refMode;
    }

    std::shared_ptr<InferenceEngine::Core> getCore() {
        return core;
    }

    virtual void ConfigureNetwork();

    virtual void LoadNetwork();

    virtual void GenerateInputs();

    virtual void Infer();

    TargetDevice targetDevice;
    std::shared_ptr<ngraph::Function> function;
    std::map<std::string, std::string> configuration;
    // Non default values of layouts/precisions will be set to CNNNetwork
    InferenceEngine::Layout inLayout = InferenceEngine::Layout::ANY;
    InferenceEngine::Layout outLayout = InferenceEngine::Layout::ANY;
    InferenceEngine::Precision inPrc = InferenceEngine::Precision::UNSPECIFIED;
    InferenceEngine::Precision outPrc = InferenceEngine::Precision::UNSPECIFIED;
    InferenceEngine::ExecutableNetwork executableNetwork;
    std::vector<InferenceEngine::Blob::Ptr> inputs;
    float threshold;
    InferenceEngine::CNNNetwork cnnNetwork;
    std::shared_ptr<InferenceEngine::Core> core;

    virtual void Validate();

    virtual std::vector<std::vector<std::uint8_t>> CalculateRefs();

    virtual std::vector<InferenceEngine::Blob::Ptr> GetOutputs();

    InferenceEngine::InferRequest inferRequest;

private:
    RefMode refMode = RefMode::INTERPRETER;
};

}  // namespace LayerTestsUtils

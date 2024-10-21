// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/functional.h>
#include "openvino/genai/text2image/pipeline.hpp"
#include "py_utils.hpp"

namespace py = pybind11;
namespace utils = ov::genai::pybind::utils;

namespace ov {
namespace genai {

/// Trampoline class to support inheritance from Generator in Python
class PyGenerator : public ov::genai::Generator {
    public:
    using ov::genai::Generator::Generator;

    float next() override {
        PYBIND11_OVERRIDE_PURE(float, Generator, next);
    }
};
}
}


auto text2image_generate_docstring = R"(
    Generates images for text-to-image models.

    :param prompt: input prompt
    :type prompt: str

    :param kwargs: arbitrary keyword arguments with keys corresponding to generate params. 
    
    Expected parameters list: 
    prompt_2: str - second prompt,
    prompt_3: str - third prompt,
    negative_prompt: str - negative prompt,
    negative_prompt_2: str - second negative prompt,
    negative_prompt_3: str - third negative prompt,
    num_images_per_prompt: int - number of images, that should be generated per prompt,
    guidance_scale: float - guidance scale,
    generation_config: GenerationConfig,
    height: int - height of resulting images,
    width: int - width of resulting images,
    num_inference_steps: int - number of inference steps,
    random_generator: openvino_genai.CppStdGenerator or class inherited from openvino_genai.Generator - random generator

    :return: ov.Tensor with resulting images
    :rtype: ov.Tensor
)";


void update_text2image_config_from_kwargs(
    ov::genai::Text2ImagePipeline::GenerationConfig& config,
    const py::kwargs& kwargs) {
    for (const auto& item : kwargs) {
        std::string key = py::cast<std::string>(item.first);
        py::object value = py::cast<py::object>(item.second);

        if (key == "prompt_2") {
            config.prompt_2 = py::cast<std::string>(value);
        } else if (key == "prompt_3") {
            config.prompt_3 = py::cast<std::string>(value);
        } else if (key == "negative_prompt") {
            config.negative_prompt = py::cast<std::string>(value);
        } else if (key == "negative_prompt_2") {
            config.negative_prompt_2 = py::cast<std::string>(value);
        } else if (key == "negative_prompt_3") {
            config.negative_prompt_3 = py::cast<std::string>(value);
        } else if (key == "num_images_per_prompt") {
            config.num_images_per_prompt = py::cast<size_t>(value);
        } else if (key == "guidance_scale") {
            config.guidance_scale = py::cast<float>(value);
        } else if (key == "height") {
            config.height = py::cast<int64_t>(value);
        } else if (key == "width") {
            config.width = py::cast<int64_t>(value);
        } else if (key == "num_inference_steps") {
            config.num_inference_steps = py::cast<size_t>(value);
        } else if (key == "random_generator") {
            auto py_generator =py::cast<std::shared_ptr<ov::genai::Generator>>(value);
            config.random_generator = py_generator;
        } else if (key == "adapters") {
            config.adapters = py::cast<ov::genai::AdapterConfig>(value);
        } else {
            throw(std::invalid_argument("'" + key + "' is unexpected parameter name. "
                                        "Use help(openvino_genai.Text2ImagePipeline.GenerationConfig) to get list of acceptable parameters."));
        }
    }
}

ov::AnyMap text2image_kwargs_to_any_map(const py::kwargs& kwargs, bool allow_compile_properties=true) {
    ov::AnyMap params = {};

    for (const auto& item : kwargs) {
        std::string key = py::cast<std::string>(item.first);
        py::object value = py::cast<py::object>(item.second);

        if (key == "prompt_2") {
            params.insert({ov::genai::prompt_2(std::move(py::cast<std::string>(value)))});
        } else if (key == "prompt_3") {
            params.insert({ov::genai::prompt_3(std::move(py::cast<std::string>(value)))});
        } else if (key == "negative_prompt") {
            params.insert({ov::genai::negative_prompt(std::move(py::cast<std::string>(value)))});
        } else if (key == "negative_prompt_2") {
            params.insert({ov::genai::negative_prompt_2(std::move(py::cast<std::string>(value)))});
        } else if (key == "negative_prompt_3") {
            params.insert({ov::genai::negative_prompt_3(std::move(py::cast<std::string>(value)))});
        } else if (key == "num_images_per_prompt") {
            params.insert({ov::genai::num_images_per_prompt(std::move(py::cast<size_t>(value)))});
        } else if (key == "guidance_scale") {
            params.insert({ov::genai::guidance_scale(std::move(py::cast<float>(value)))});
        } else if (key == "height") {
            params.insert({ov::genai::height(std::move(py::cast<int64_t>(value)))});
        } else if (key == "width") {
            params.insert({ov::genai::width(std::move(py::cast<int64_t>(value)))});
        } else if (key == "num_inference_steps") {
            params.insert({ov::genai::num_inference_steps(std::move(py::cast<size_t>(value)))});
        } else if (key == "random_generator") {
            auto py_generator =py::cast<std::shared_ptr<ov::genai::Generator>>(value);
            params.insert({ov::genai::random_generator(std::move(py_generator))});
        } else if (key == "adapters") {
            params.insert({ov::genai::adapters(std::move(py::cast<ov::genai::AdapterConfig>(value)))});
        }
        else {
            if (allow_compile_properties) {
                // convert arbitrary objects to ov::Any
                // not supported properties are not checked, as these properties are passed to compile(), which will throw exception in case of unsupported property
                if (utils::py_object_is_any_map(value)) {
                    auto map = utils::py_object_to_any_map(value);
                    params.insert(map.begin(), map.end());
                } else {
                    params[key] = utils::py_object_to_any(value);
                }
            }
            else {
                // generate doesn't run compile(), so only Text2ImagePipeline specific properties are allowed
                throw(std::invalid_argument("'" + key + "' is unexpected parameter name. "
                                            "Use help(openvino_genai.Text2ImagePipeline.generate) to get list of acceptable parameters."));
            }
        }
        
        
    }
    return params;
}

void init_text2image_pipeline(py::module_& m) {

    py::class_<ov::genai::Generator, ov::genai::PyGenerator, std::shared_ptr<ov::genai::Generator>>(m, "Generator", "This class is used for storing pseudo-random generator.")
        .def(py::init<>());
    
    py::class_<ov::genai::CppStdGenerator, ov::genai::Generator, std::shared_ptr<ov::genai::CppStdGenerator>>(m, "CppStdGenerator", "This class wraps std::mt19937 pseudo-random generator.")
        .def(py::init([](
            uint32_t seed
        ) {
            return std::make_unique<ov::genai::CppStdGenerator>(seed);
        }))
        .def("next", &ov::genai::CppStdGenerator::next);


    auto text2image_pipeline = py::class_<ov::genai::Text2ImagePipeline>(m, "Text2ImagePipeline", "This class is used for generation with text-to-image models.")
        .def(py::init([](
            const std::string& model_path
        ) {
            return std::make_unique<ov::genai::Text2ImagePipeline>(model_path);
        }),
        py::arg("model_path"), "folder with exported model files.", 
        R"(
            Text2ImagePipeline class constructor.
            model_path (str): Path to the folder with exported model files.
        )")

        .def(py::init([](
            const std::string& model_path, 
            const std::string& device,
            const py::kwargs& kwargs
        ) {
            return std::make_unique<ov::genai::Text2ImagePipeline>(model_path, device, text2image_kwargs_to_any_map(kwargs, true));
        }),
        py::arg("model_path"), "folder with exported model files.", 
        py::arg("device") = "CPU", "device on which inference will be done",
        R"(
            Text2ImagePipeline class constructor.
            model_path (str): Path with exported model files.
            device (str): Device to run the model on (e.g., CPU, GPU).
            kwargs: Text2ImagePipeline properties
        )")
        .def("get_generation_config", &ov::genai::Text2ImagePipeline::get_generation_config)
        .def("set_generation_config", &ov::genai::Text2ImagePipeline::set_generation_config)
        .def("set_scheduler", &ov::genai::Text2ImagePipeline::set_scheduler)
        .def("reshape", &ov::genai::Text2ImagePipeline::reshape)
        .def("stable_diffusion", &ov::genai::Text2ImagePipeline::stable_diffusion)
        .def("latent_consistency_model", &ov::genai::Text2ImagePipeline::latent_consistency_model)
        .def("stable_diffusion_xl", &ov::genai::Text2ImagePipeline::stable_diffusion_xl)
        .def(
            "compile", 
            [](ov::genai::Text2ImagePipeline& pipe, 
                const std::string& device,
                const py::kwargs& kwargs
            ) {
                pipe.compile(device,  utils::kwargs_to_any_map(kwargs));
            },
            py::arg("device") = "CPU", "device on which inference will be done",
            R"(
                Compiles the model.
                device (str): Device to run the model on (e.g., CPU, GPU).
                kwargs: Device properties.
            )")
        .def(
            "generate", 
            [](ov::genai::Text2ImagePipeline& pipe, 
                const std::string& prompt,
                const py::kwargs& kwargs
            ) {
                ov::AnyMap params = text2image_kwargs_to_any_map(kwargs, false);
                return py::cast(pipe.generate(prompt, params));
            },
            py::arg("prompt"), "Input string",
            (text2image_generate_docstring + std::string(" \n ")).c_str()
        );
    
    auto text2image_scheduler = py::class_<ov::genai::Text2ImagePipeline::Scheduler>(text2image_pipeline, "Scheduler", "This class is used for storing scheduler configuration for Text2Image pipeline.")
        .def(py::init<>());

    py::enum_<ov::genai::Text2ImagePipeline::Scheduler::Type>(text2image_scheduler, "Type")
        .value("AUTO", ov::genai::Text2ImagePipeline::Scheduler::Type::AUTO)
        .value("LCM", ov::genai::Text2ImagePipeline::Scheduler::Type::LCM)
        .value("LMS_DISCRETE", ov::genai::Text2ImagePipeline::Scheduler::Type::LMS_DISCRETE)
        .value("DDIM", ov::genai::Text2ImagePipeline::Scheduler::Type::DDIM)
        .value("EULER_DISCRETE", ov::genai::Text2ImagePipeline::Scheduler::Type::EULER_DISCRETE);
    
    text2image_scheduler.def("from_config", &ov::genai::Text2ImagePipeline::Scheduler::from_config);

    auto text2image_generation_config = py::class_<ov::genai::Text2ImagePipeline::GenerationConfig>(text2image_pipeline, "GenerationConfig", "This class is used for storing generation config for Text2Image pipeline.")
        .def(py::init<>())
        .def_readwrite("prompt_2", &ov::genai::Text2ImagePipeline::GenerationConfig::prompt_2)
        .def_readwrite("prompt_3", &ov::genai::Text2ImagePipeline::GenerationConfig::prompt_3)
        .def_readwrite("negative_prompt", &ov::genai::Text2ImagePipeline::GenerationConfig::negative_prompt)
        .def_readwrite("negative_prompt_2", &ov::genai::Text2ImagePipeline::GenerationConfig::negative_prompt_2)
        .def_readwrite("negative_prompt_3", &ov::genai::Text2ImagePipeline::GenerationConfig::negative_prompt_3)
        .def_readwrite("random_generator", &ov::genai::Text2ImagePipeline::GenerationConfig::random_generator)
        .def_readwrite("guidance_scale", &ov::genai::Text2ImagePipeline::GenerationConfig::guidance_scale)
        .def_readwrite("height", &ov::genai::Text2ImagePipeline::GenerationConfig::height)
        .def_readwrite("width", &ov::genai::Text2ImagePipeline::GenerationConfig::width)
        .def_readwrite("num_inference_steps", &ov::genai::Text2ImagePipeline::GenerationConfig::num_inference_steps)
        .def_readwrite("num_images_per_prompt", &ov::genai::Text2ImagePipeline::GenerationConfig::num_images_per_prompt)
        .def_readwrite("adapters", &ov::genai::Text2ImagePipeline::GenerationConfig::adapters)

        .def("validate", &ov::genai::Text2ImagePipeline::GenerationConfig::validate)

        .def("update_generation_config", [](
        ov::genai::Text2ImagePipeline::GenerationConfig config, 
        const py::kwargs& kwargs) {
            update_text2image_config_from_kwargs(config, kwargs);
        });

}
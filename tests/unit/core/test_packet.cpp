// tests/unit/core/test_packet.cpp
#include <gtest/gtest.h>
#include "ai_stream/core/packet.h"

using namespace ai_stream;
using namespace ai_stream::core;

TEST(BasePacketTest, DefaultValuesInitialized) {
    BasePacket p;
    EXPECT_EQ(p.type, PacketType::UNKNOWN);
    EXPECT_EQ(p.timestamp_ms, 0);
    EXPECT_EQ(p.stream_id, 0u);
    EXPECT_EQ(p.frame_id, 0);
    EXPECT_EQ(p.cost_ms, 0u);
    EXPECT_TRUE(p.source_id.empty());
    EXPECT_TRUE(p.cost_time_map.empty());
}

TEST(PacketTypeTest, DerivedPacketTypes) {
    RawVideoPacket raw;
    EXPECT_EQ(raw.type, PacketType::RAW_VIDEO);

    VideoFramePacket frame;
    EXPECT_EQ(frame.type, PacketType::DECODED_FRAME);
    EXPECT_EQ(frame.channels, 3);
    EXPECT_FALSE(frame.is_gpu);
    EXPECT_FALSE(frame.letterbox_used);

    InferenceResultPacket infer;
    EXPECT_EQ(infer.type, PacketType::META_DATA);
}

TEST(AlertEventTest, ToJsonContainsAllFields) {
    rules::AlertEvent e;
    e.alert_id = "alert-001";
    e.alert_name = "smoking";
    e.alert_type = rules::AlertType::SMOKING;
    e.alert_item_type = rules::AlertItemType::ITEM_PERSON_BEHAVIOR;
    e.level = rules::AlertLevel::WARNING;
    e.status = rules::AlertStatus::ALERT_STATUS_OCCUR;
    e.description = "smoking occur";
    e.detect_ms = 1000;
    e.duration_ms = 2500;
    e.non_update_count = 3;
    e.zone_no = 2;
    e.object_ids = {10, 20};
    e.extra_data = {{"score", 0.9}};

    auto j = e.toJson();
    EXPECT_EQ(j["alert_id"], "alert-001");
    EXPECT_EQ(j["alert_name"], "smoking");
    EXPECT_EQ(j["alert_type"], static_cast<int>(rules::AlertType::SMOKING));
    EXPECT_EQ(j["alert_type_name"], "smoking");
    EXPECT_EQ(j["alert_item_type"], static_cast<int>(rules::AlertItemType::ITEM_PERSON_BEHAVIOR));
    EXPECT_EQ(j["level"], static_cast<int>(rules::AlertLevel::WARNING));
    EXPECT_EQ(j["level_name"], "warning");
    EXPECT_EQ(j["status"], static_cast<int>(rules::AlertStatus::ALERT_STATUS_OCCUR));
    EXPECT_EQ(j["status_name"], "occur");
    EXPECT_EQ(j["description"], "smoking occur");
    EXPECT_EQ(j["detect_ms"], 1000);
    EXPECT_EQ(j["duration_ms"], 2500);
    EXPECT_EQ(j["non_update_count"], 3);
    EXPECT_EQ(j["zone_no"], 2);
    ASSERT_EQ(j["object_ids"].size(), 2u);
    EXPECT_EQ(j["object_ids"][0], 10);
    EXPECT_EQ(j["extra_data"]["score"], 0.9);
}

TEST(AlertEventTest, ToJsonDefaultsAndOptionalFields) {
    rules::AlertEvent e;
    auto j = e.toJson();
    EXPECT_EQ(j["alert_type"], static_cast<int>(rules::AlertType::ALERT_UNKNOWN));
    EXPECT_EQ(j["alert_item_type"], static_cast<int>(rules::AlertItemType::ITEM_MASTER_UNKNOWN));
    EXPECT_EQ(j["status_name"], "default");
    // extra_data 为 null 时不输出
    EXPECT_FALSE(j.contains("extra_data"));
}

TEST(AlertTypeMapTest, AllMapsContainActionRecognition) {
    EXPECT_TRUE(rules::alertTypeMap.count(rules::AlertType::ACTION_RECOGNITION) == 1);
    EXPECT_EQ(rules::alertTypeMap.at(rules::AlertType::ACTION_RECOGNITION), "action_recognition");

    EXPECT_TRUE(rules::alertTypeChMap.count(rules::AlertType::ACTION_RECOGNITION) == 1);

    EXPECT_TRUE(rules::alertItemTypeMap.count(rules::AlertType::ACTION_RECOGNITION) == 1);
    EXPECT_EQ(rules::alertItemTypeMap.at(rules::AlertType::ACTION_RECOGNITION),
              rules::AlertItemType::ITEM_PERSON_BEHAVIOR);
}

TEST(AlertTypeMapTest, AllAlertTypesHaveMappings) {
    // 除 ALERT_UNKNOWN 外的每个枚举值都应在三个映射表中有条目
    std::vector<rules::AlertType> types = {
        rules::AlertType::PERSON_INTRUSION,
        rules::AlertType::MISSING_HELMET,
        rules::AlertType::MISSING_WORK_CLOTHES,
        rules::AlertType::PHONE_CALL,
        rules::AlertType::SMOKING,
        rules::AlertType::FALL_DOWN,
        rules::AlertType::MISSING_SAFETY_BELT,
        rules::AlertType::HUMAN_GATHERING,
        rules::AlertType::ABSENCE,
        rules::AlertType::SLEEPING_ON_DUTY,
        rules::AlertType::CLIMBING,
        rules::AlertType::FIGHTING,
        rules::AlertType::UNLICENSED_VENDOR,
        rules::AlertType::PHOTOGRAPHER,
        rules::AlertType::FIRE_LANE_OCCUPANCY,
        rules::AlertType::DISCOVER_CRYSTAL,
        rules::AlertType::DISCOVER_VISIBLE_FIRE,
        rules::AlertType::DISCOVER_SMOKE,
        rules::AlertType::DISCOVER_HOSE_CUTOFF,
        rules::AlertType::ACTION_RECOGNITION,
    };
    for (auto t : types) {
        EXPECT_EQ(rules::alertTypeMap.count(t), 1u) << "missing in alertTypeMap";
        EXPECT_EQ(rules::alertTypeChMap.count(t), 1u) << "missing in alertTypeChMap";
        EXPECT_EQ(rules::alertItemTypeMap.count(t), 1u) << "missing in alertItemTypeMap";
    }
}

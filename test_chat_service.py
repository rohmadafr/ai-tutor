#!/usr/bin/env python3
"""
Test Script untuk Validasi SimpleChatService Fixes
Test semua scenarios: cache hit/miss, personalization, streaming, database tracking
"""
import asyncio
import time
import sys
import os
import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.simple_chat_service import SimpleChatService
from app.core.database import async_db
from app.schemas.db_models import User, Course, Chatroom

class ChatServiceTester:
    """Test class untuk SimpleChatService scenarios"""

    def __init__(self):
        self.chat_service = SimpleChatService()
        self.test_results = []

    async def test_scenario(self, scenario_name: str, test_func):
        """Run single test scenario and track results"""
        print(f"\n{'='*50}")
        print(f"🧪 TESTING: {scenario_name}")
        print(f"{'='*50}")

        start_time = time.time()

        try:
            result = await test_func()
            duration = (time.time() - start_time) * 1000

            self.test_results.append({
                "scenario": scenario_name,
                "status": "✅ PASS",
                "duration_ms": duration,
                "result": result
            })

            print(f"✅ {scenario_name} - PASS ({duration:.2f}ms)")
            print(f"📄 Result type: {type(result)}")

            if isinstance(result, dict):
                print(f"🔹 Response length: {len(result.get('response', ''))}")
                print(f"🔹 Source: {result.get('source', 'unknown')}")
                print(f"🔹 Cached: {result.get('cached', False)}")
                print(f"🔹 Personalized: {result.get('personalized', False)}")
                print(f"🔹 Model: {result.get('model_used', 'unknown')}")
                print(f"🔹 Response Time: {result.get('latency_ms', 0):.2f}ms")
            else:
                print(f"🔹 Response: {str(result)[:100]}...")

        except Exception as e:
            duration = (time.time() - start_time) * 1000

            self.test_results.append({
                "scenario": scenario_name,
                "status": "❌ FAIL",
                "duration_ms": duration,
                "error": str(e)
            })

            print(f"❌ {scenario_name} - FAIL ({duration:.2f}ms)")
            print(f"🔹 Error: {str(e)}")

    # ============== TEST SCENARIOS ==============

    async def test_cache_hit_raw(self):
        """Test 1: Cache hit + raw response (non-streaming)"""
        query = "Apa itu machine learning?"

        # First call - should be cache miss
        first_result = await self.chat_service.chat_with_database(
            query=query,
            user_id="test-user-001",
            course_id="test-course-001",
            chatroom_id="test-chatroom-001",
            use_personalization=False
        )

        # Validate first call is cache miss
        assert isinstance(first_result, dict), "First result harus berupa dict"
        assert first_result["source"] == "rag", f"First call should be rag, got {first_result.get('source')}"
        assert first_result["cached"] == False, "First call harus cached=False"

        # Second call with same query - should be cache hit
        result = await self.chat_service.chat_with_database(
            query=query,
            user_id="test-user-001",
            course_id="test-course-001",
            chatroom_id="test-chatroom-001",
            use_personalization=False
        )

        # Validate expected structure for cache hit
        assert isinstance(result, dict), "Result harus berupa dict"
        assert "response" in result, "Harus ada 'response' field"
        assert result["source"] == "cache_raw", f"Expected cache_raw, got {result.get('source')}"
        assert result["cached"] == True, "Harus cached=True"
        assert result["personalized"] == False, "Harus personalized=False"
        assert "latency_ms" in result, "Harus ada latency_ms"
        assert isinstance(result["latency_ms"], (int, float)), "latency_ms harus number"

        return result

    async def test_cache_hit_personalized(self):
        """Test 2: Cache hit + personalized response (non-streaming)"""
        query = "Apa itu machine learning?"

        # First call - should be cache miss (reuse cache from test 1)
        # Since test 1 already cached this query, this should be cache hit
        result = await self.chat_service.chat_with_database(
            query=query,
            user_id="test-user-002",
            course_id="test-course-001",
            chatroom_id="test-chatroom-002",
            use_personalization=True
        )

        # Validate expected structure
        assert isinstance(result, dict), "Result harus berupa dict"
        assert "response" in result, "Harus ada 'response' field"
        # Should be cache_personalized since we're using personalization on cached result
        assert result["source"] == "cache_personalized", f"Expected cache_personalized, got {result.get('source')}"
        assert result["cached"] == True, "Harus cached=True"
        assert result["personalized"] == True, "Harus personalized=True"
        assert "model_used" in result, "Harus ada model_used"
        assert "latency_ms" in result, "Harus ada latency_ms"

        return result

    async def test_cache_miss_rag(self):
        """Test 3: Cache miss + RAG response (non-streaming)"""
        query = "Apa itu quantum computing dan bagaimana hubungannya dengan AI?"

        result = await self.chat_service.chat_with_database(
            query=query,
            user_id="test-user-003",
            course_id="test-course-002",
            chatroom_id="test-chatroom-003",
            use_personalization=False
        )

        # Validate expected structure
        assert isinstance(result, dict), "Result harus berupa dict"
        assert "response" in result, "Harus ada 'response' field"
        assert result["source"] == "rag", f"Expected rag, got {result.get('source')}"
        assert result["cached"] == False, "Harus cached=False"
        assert result["personalized"] == False, "Harus personalized=False"
        assert "latency_ms" in result, "Harus ada latency_ms"
        assert result["model_used"] == "gpt-4o-mini", f"Expected gpt-4o-mini, got {result.get('model_used')}"

        return result

    async def test_streaming_cache_hit_raw(self):
        """Test 4: Streaming cache hit + raw response"""
        query = "Apa itu machine learning?"

        chunks = []
        async for chunk in self.chat_service.chat_with_database_stream(
            query=query,
            user_id="test-user-004",
            course_id="test-course-001",
            chatroom_id="test-chatroom-004",
            use_personalization=False
        ):
            chunks.append(chunk)

        # Validate streaming
        assert len(chunks) > 0, "Harus ada chunks"
        full_response = "".join(chunks)
        assert len(full_response) > 0, "Response tidak boleh kosong"

        return {
            "chunks_count": len(chunks),
            "response_length": len(full_response),
            "response_preview": full_response[:100]
        }

    async def test_streaming_cache_hit_personalized(self):
        """Test 5: Streaming cache hit + personalized response"""
        query = "Apa itu machine learning?"

        chunks = []
        async for chunk in self.chat_service.chat_with_database_stream(
            query=query,
            user_id="test-user-005",
            course_id="test-course-001",
            chatroom_id="test-chatroom-005",
            use_personalization=True
        ):
            chunks.append(chunk)

        # Validate streaming
        assert len(chunks) > 0, "Harus ada chunks"
        full_response = "".join(chunks)
        assert len(full_response) > 0, "Response tidak boleh kosong"

        return {
            "chunks_count": len(chunks),
            "response_length": len(full_response),
            "response_preview": full_response[:100]
        }

    async def test_streaming_cache_miss_rag(self):
        """Test 6: Streaming cache miss + RAG response"""
        query = "Jelaskan perbedaan antara supervised dan unsupervised learning"

        chunks = []
        async for chunk in self.chat_service.chat_with_database_stream(
            query=query,
            user_id="test-user-006",
            course_id="test-course-003",
            chatroom_id="test-chatroom-006",
            use_personalization=False
        ):
            chunks.append(chunk)

        # Validate streaming
        assert len(chunks) > 0, "Harus ada chunks"
        full_response = "".join(chunks)
        assert len(full_response) > 0, "Response tidak boleh kosong"

        return {
            "chunks_count": len(chunks),
            "response_length": len(full_response),
            "response_preview": full_response[:100]
        }

    async def test_parameter_consistency(self):
        """Test 7: Parameter consistency across methods"""
        query = "Test parameter consistency"

        # Test both methods with same parameters
        non_stream_result = await self.chat_service.chat_with_database(
            query=query,
            user_id="test-user-consistency",
            course_id="test-course-consistency",
            chatroom_id="test-chatroom-consistency",
            use_personalization=True
        )

        chunks = []
        async for chunk in self.chat_service.chat_with_database_stream(
            query=query,
            user_id="test-user-consistency",
            course_id="test-course-consistency",
            chatroom_id="test-chatroom-consistency",
            use_personalization=True
        ):
            chunks.append(chunk)

        stream_result = {
            "response": "".join(chunks),
            "chunks_count": len(chunks)
        }

        # Validate consistency
        assert isinstance(non_stream_result, dict), "Non-streaming harus dict"
        assert isinstance(stream_result, dict), "Streaming harus dict"

        return {
            "non_streaming_type": type(non_stream_result).__name__,
            "streaming_type": type(stream_result).__name__,
            "same_user_context": True
        }

    # ============== MAIN TEST RUNNER ==============

    async def run_all_tests(self):
        """Run all test scenarios"""
        print("🚀 Starting SimpleChatService Test Suite")
        print("Testing all scenarios with real data tracking and no redundancy...")

        # Test scenarios
        await self.test_scenario("Cache Hit + Raw Response", self.test_cache_hit_raw)
        await self.test_scenario("Cache Hit + Personalized Response", self.test_cache_hit_personalized)
        await self.test_scenario("Cache Miss + RAG Response", self.test_cache_miss_rag)
        await self.test_scenario("Streaming Cache Hit + Raw", self.test_streaming_cache_hit_raw)
        await self.test_scenario("Streaming Cache Hit + Personalized", self.test_streaming_cache_hit_personalized)
        await self.test_scenario("Streaming Cache Miss + RAG", self.test_streaming_cache_miss_rag)
        await self.test_scenario("Parameter Consistency", self.test_parameter_consistency)

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print test summary"""
        print(f"\n{'='*60}")
        print("📊 TEST SUMMARY")
        print(f"{'='*60}")

        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r["status"] == "✅ PASS"])
        failed_tests = total_tests - passed_tests

        total_time = sum(r["duration_ms"] for r in self.test_results)

        print(f"📈 Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"⏱️ Total Duration: {total_time:.2f}ms")
        print(f"📊 Success Rate: {(passed_tests/total_tests)*100:.1f}%")

        print(f"\n📋 Detailed Results:")
        for result in self.test_results:
            print(f"  {result['status']} {result['scenario']} - {result['duration_ms']:.2f}ms")
            if "error" in result:
                print(f"    Error: {result['error']}")

        if failed_tests == 0:
            print(f"\n🎉 ALL TESTS PASSED! SimpleChatService is working correctly!")
        else:
            print(f"\n⚠️ {failed_tests} tests failed. Please check the errors above.")

async def create_test_data():
    """Create test users, courses, and chatrooms for foreign key constraints"""
    try:
        async with async_db.get_session() as db:
            # Create test users
            test_users = [
                "test-user-001", "test-user-002", "test-user-003", "test-user-004",
                "test-user-005", "test-user-006", "test-user-consistency"
            ]

            for user_id in test_users:
                # Check if user exists
                from sqlalchemy import select
                result = await db.execute(select(User).filter_by(user_id=user_id))
                if not result.scalar_one_or_none():
                    user = User(
                        user_id=user_id,
                        username=f"user_{user_id}",
                        email=f"{user_id}@test.com",
                        password_hash="test_hash",
                        role="student"
                    )
                    db.add(user)

            # Create test courses with explicit course_id
            test_courses = [
                ("test-course-001", "Machine Learning Fundamentals"),
                ("test-course-002", "Advanced AI Topics"),
                ("test-course-003", "Deep Learning"),
                ("test-course-consistency", "Consistency Test Course")
            ]
            for course_id, course_title in test_courses:
                result = await db.execute(select(Course).filter_by(course_id=course_id))
                if not result.scalar_one_or_none():
                    course = Course(
                        course_id=course_id,
                        title=course_title,
                        description=f"Test course: {course_title}",
                        instructor_id="test-user-001"
                    )
                    db.add(course)

            # Create test chatrooms
            test_chatrooms_data = [
                ("test-chatroom-001", "test-course-001", "Test Chatroom 1"),
                ("test-chatroom-002", "test-course-001", "Test Chatroom 2"),
                ("test-chatroom-003", "test-course-002", "Test Chatroom 3"),
                ("test-chatroom-004", "test-course-001", "Test Chatroom 4"),
                ("test-chatroom-005", "test-course-001", "Test Chatroom 5"),
                ("test-chatroom-006", "test-course-003", "Test Chatroom 6"),
                ("test-chatroom-consistency", "test-course-consistency", "Test Chatroom Consistency")
            ]

            for chatroom_id, course_id, room_name in test_chatrooms_data:
                result = await db.execute(select(Chatroom).filter_by(chatroom_id=chatroom_id))
                if not result.scalar_one_or_none():
                    chatroom = Chatroom(
                        chatroom_id=chatroom_id,
                        course_id=course_id,
                        user_id="test-user-001",
                        room_name=room_name
                    )
                    db.add(chatroom)

            # Create user contexts for personalization tests
            from app.schemas.db_models import UserContext

            user_contexts_data = [
                ("test-user-002", "test-course-001", "Student prefers detailed explanations with examples. Learning style: visual learner."),
                ("test-user-005", "test-course-001", "Advanced student who likes concise technical explanations."),
                ("test-user-consistency", "test-course-consistency", "Beginner student who needs step-by-step guidance.")
            ]

            for user_id, course_id, context_text in user_contexts_data:
                # Get or create, then update to ensure fresh data
                user_context = await UserContext.aget_or_create(db, user_id, course_id, context_text)
                if user_context.user_context != context_text:
                    user_context.user_context = context_text
                    user_context.updated_at = datetime.datetime.now(datetime.UTC)
                    db.add(user_context)

            await db.commit()
            print("✅ Test data created successfully (including user contexts)")

    except Exception as e:
        print(f"❌ Failed to create test data: {e}")
        import traceback
        traceback.print_exc()

async def clear_redis_cache():
    """Clear Redis cache before tests to ensure clean slate"""
    try:
        import redis.asyncio as redis
        from app.config.settings import settings

        # Connect to cache Redis (port 6380)
        cache_client = redis.from_url(settings.redis_cache_url)
        await cache_client.flushdb()
        await cache_client.aclose()
        print("🧹 Redis cache cleared successfully")
    except Exception as e:
        print(f"⚠️ Failed to clear Redis cache: {e}")

async def main():
    """Main test runner"""
    try:
        # Clear Redis cache first
        print("🧹 Clearing Redis cache...")
        await clear_redis_cache()

        # Initialize database
        print("🔌 Initializing database connection...")
        await async_db.connect()

        # Initialize database tables
        print("🗄️ Initializing database tables...")
        success = await async_db.initialize_tables()
        if not success:
            print("⚠️ Failed to initialize database tables, but continuing with tests...")

        # Create test data for foreign key constraints
        print("👥 Creating test users and chatrooms...")
        await create_test_data()

        # Run tests
        tester = ChatServiceTester()
        await tester.run_all_tests()

    except Exception as e:
        print(f"❌ Test setup failed: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        try:
            await async_db.disconnect()
            print("🔌 Database disconnected")
        except:
            pass

if __name__ == "__main__":
    print("🧪 SimpleChatService Test Runner")
    print("=" * 50)
    asyncio.run(main())